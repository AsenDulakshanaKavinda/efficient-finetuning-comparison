import time
import torch
from contextlib import contextmanager
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

def count_params(model: AutoModelForSequenceClassification)-> tuple[int, int, float]:
    """ Return (trainable params, total params, trainable params percentage)"""

    # count of all the trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # count of all the parameters (both trainable and un trainable)
    total = sum(p.numel() for p in model.parameters())

    # percentage of trainable parameters
    trainable_params_percentage = 100 * trainable / total if total > 0 else 0.0

    return trainable, total, trainable_params_percentage


def compute_f1(predictions, labels) -> tuple[float, float]:
    """
    Predictions: raw logits (numpy array), as return by trainer.predict()
    labels: ground truth label ids
    Return: (f1, accuracy). use macro F1 so class imbalance doesn't hide a model that only predicts the majority class
    """

    preds = predictions.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)

    return f1, accuracy


@contextmanager
def track_peak_memory():
    """
    Usage:
        with track_peak_memory() as mem:
            trainer.train()
        print(mem["peak_gb"])

    Must be used on CUDA. resets the peak counter on entry so results aren't contaminated by whatever ran before this block.
    """
    if not torch.cuda.is_available():
        # CPU fallback so the script doesn't crash in a CPU-only sanity check
        result = {"peak_gb": 0.0}
        yield result
        return

    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.empty_cache()
    result = {}
    try:
        yield result
    finally:
        result["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9


def benchmark_latency(model, dataset, num_samples = 50, num_warmup = 4, device = None):

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    samples = dataset.select(range(min(num_samples + num_warmup, len(dataset))))

    times = [] 
    with torch.no_grad():
        for i, example in enumerate(samples):
            batch = {
                k: torch.tensor(v).unsqueeze(0).to(device)
                for k, v in example.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(**batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            if i >= num_warmup:
                time.append(elapsed_ms)

    return sum(times) / len(times)



