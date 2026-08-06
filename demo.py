import yaml
from pydantic import BaseModel


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


def load_config(path = None):
    """read config from given yaml file"""
    try:
        print("Reading configuration for full feature training and validating")
        path = "./config/full_ft.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        return FullFeatureTraining(**config)
    except Exception as e:
        print(f"Error while loading full feature training config: {e!s}")
        raise Exception(e) from e

c = load_config()
print(c.seed)