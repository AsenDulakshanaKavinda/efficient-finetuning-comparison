import argparse
import time

import mlflow
import torch
import yaml
from pydantic import BaseModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data_prep import DataProcessing
from src.metrics_utils import (
    benchmark_latency,
    compute_f1,
    count_params,
    track_peak_memory,
)
from utils import get_logger

log = get_logger(__file__)

class FullFeatureTraining(BaseModel):
    num_labels: int
    max_length: int
    seed: int
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float       
    output_dir: str
    mlflow_experiment_name: str


def load_config():
    """read config from given yaml file"""
    try:
        log.info("Reading configuration for full feature training and validating")
        path = "config/full_ft.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        return FullFeatureTraining(**config)
    except Exception as e:
        log.error(f"Error while loading full feature training config: {e!s}")
        raise Exception(e) from e

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/full_ft.yaml")
    return p.parse_args()


def build_model_full(cfg: FullFeatureTraining):
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=cfg.model_name, 
        num_labels=cfg.num_labels
    )
    return model


def main():
    args = parse_args()
    cfg = load_config(args.config)


    torch.manual_seed(cfg.seed)

    # --- Setup experiment ---
    mlflow.set_experiment(
        experiment_name=cfg.mlflow_experiment_name
    )

    
    with mlflow.start_run(run_name="full features"):
        mlflow.log_params(
            {
                "method": "full features",
                "model_name": cfg.model_name,
                "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate,
                "batch_size": cfg.batch_size,
                "seed": cfg.seed,
            }
        )

        # --- Setup Model and Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model = build_model_full(cfg=cfg)

        trainable, total, pct = count_params(model)

        # --- Setup data ----
        data_processor = DataProcessing()
        raw_dataset = data_processor.load_dataset()
        train_data, validation_data, test_data = data_processor.split_dataset(
            dataset = raw_dataset,
            tokenizer = tokenizer
        )

        mlflow.log_metrics(
            {"trainable_params": trainable, "total_params": total, "trainable_pct": pct}
        )

        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.3f}%)")

        # --- Training arguments ---
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
            report_to=[],
        )

        # -- Trainer ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=validation_data,
            processing_class=tokenizer,
        )

        # --- Training time + peak memory --_
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

        # Save only the adapter weights, not the full model — this is the
        # storage-efficiency story you want in the write-up.
        model.save_adapter(f"{cfg.output_dir}/full_ft", "full_ft")
        mlflow.log_artifacts(f"{cfg.output_dir}/full_ft", artifact_path="full_ft")


if __name__ == "__main__":
    main()
