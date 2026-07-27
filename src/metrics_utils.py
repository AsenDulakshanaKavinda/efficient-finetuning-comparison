
from transformers import AutoModelForSequenceClassification

def count_params(model: AutoModelForSequenceClassification)-> tuple[int, int, float]:
    # count of all the trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # count of all the parameters (both trainable and un trainable)
    total = sum(p.numel() for p in model.parameters())

    # percentage of trainable parameters
    trainable_params_percentage = (trainable / total) * 0.01

    return trainable, total, trainable_params_percentage
    ...