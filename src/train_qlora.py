import argparse
import time
import os

import mlflow
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

from src.data_prep import DataProcessing
from src.metrics_utils import (
    benchmark_latency,
    compute_f1,
    count_params,
    track_peak_memory,
)
from utils import get_logger

log = get_logger(__file__)

class QLoraTraining(BaseModel):
    dataset_name: str
    text_column: str
    label_column: str
    num_labels: int
    max_length: int
    val_split: float
    test_split: float
    seed: int
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float          
    # --- LoRA-specific ---
    lora_r: int                 
    lora_alpha: int             # commonly 2x the rank
    lora_dropout: float
    lora_target_modules: list[str]
    output_dir: str
    mlflow_experiment_name: str

def load_config(path: str) -> QLoraTraining:
    """ read config from giver yaml file. """
    try:
        log.info("Reading configuration for QLORA training and validating")
        with open(path) as f:
            config = yaml.safe_load(f)
        return QLoraTraining(**config)
    except Exception as e:
        log.error(f"Error while loading full QLORA training config: {e!s}")
        raise Exception(e) from e

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/qlora.yaml")
    return p.parse_args()

def build_model_with_qlora(cfg: QLoraTraining):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # normalized float 4 
        bnb_4bit_compute_dtype=torch.bfloat16, # compute dtype for matmul
        bnb_4bit_use_double_quant=True, # quantize the quantization constants 
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
        quantization_config=bnb_config,
        device_map="auto",  
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.seed)

    mlflow.set_experiment(cfg.mlflow_experiment_name)
    with mlflow.start_run(run_name="lora"):
        mlflow.log_params(
            {
                "method": "lora",
                "model_name": cfg.model_name,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_target_modules": cfg.lora_target_modules,
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        data_processor = DataProcessing()
        data_processor.download_dataset()
        raw_dataset = data_processor.load_dataset()
        train_data, validation_data, test_data = data_processor.split_dataset(
            dataset = raw_dataset,
            tokenizer = tokenizer
        )

        model = build_model_with_qlora(cfg)

        trainable, total, pct = count_params(model)
        mlflow.log_metrics(
            {"trainable_params": trainable, "total_params": total, "trainable_pct": pct}
        )
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")

        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            report_to=[],  # logging to MLflow manually for consistency across all 4 scripts
            fp16=False,             # Disable FP16 (or use bf16=True if supported)
            max_grad_norm=1.0,      # Prevents exploding gradients
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
        )

        # --- Training time + peak memory ---
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with track_peak_memory() as mem_tracker:
            trainer.train()
        train_time_sec = time.time() - start
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

        mlflow.log_metrics(
            {"train_time_sec": train_time_sec, "peak_gpu_mem_gb": peak_mem_gb}
        )
        print(f"Train time: {train_time_sec:.1f}s | Peak GPU mem: {peak_mem_gb:.2f} GB")

        # --- Final eval on held-out test set ---
        preds_output = trainer.predict(test_dataset=test_data)
        f1, accuracy = compute_f1(
            predictions=preds_output.predictions, labels=preds_output.label_ids
        )
        mlflow.log_metrics({"test_f1": f1, "test_accuracy": accuracy})
        print(f"Test F1: {f1:.4f} | Test Accuracy: {accuracy:.4f}")

        # --- Inference latency ---
        avg_latency_ms = benchmark_latency(model, test_data)
        mlflow.log_metric("avg_inference_latency_ms", avg_latency_ms)
        print(f"Avg inference latency: {avg_latency_ms:.2f} ms/sample")

        # --- Save the trained model ---
        save_path = f"{cfg.output_dir}/qlora_ft"
        os.makedirs(save_path, exist_ok=True)

        # Save 
        model.save_pretrained(f"{cfg.output_dir}/qlora_ft")
        tokenizer.save_pretrained(f"{cfg.output_dir}/qlora_ft")
        mlflow.log_artifacts(f"{cfg.output_dir}/qlora_ft", artifact_path="qlora_ft")





