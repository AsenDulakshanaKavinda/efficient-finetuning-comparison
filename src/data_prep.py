from pathlib import Path
import os
import shutil
import yaml
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from pydantic import BaseModel
from transformers import AutoTokenizer

from src.exceptions import DataProcessingException
from utils import get_logger

log = get_logger(__file__)


class DataConfig(BaseModel):
    dataset_repo: str
    dataset_name: str
    raw_dataset_path: str
    features: list
    target: str
    num_of_labels: int


def load_dataset_config() -> DataConfig:
    try:
        log.info("Reading configuration for Data Processing and validating")
        path = "config/dataset.yaml"
        with open(path) as f:
            configs = yaml.safe_load(f)
        return DataConfig(**configs)
    except Exception as e:
        log.error(f"Error while loading dataset config: {e!s}")
        raise Exception(e) from e


class DataProcessing:
    def __init__(self):
        self.cfg = load_dataset_config()
        self.dataset_repo: str = self.cfg.dataset_repo
        self.dataset_name: str = self.cfg.dataset_name
        self.raw_dataset_path: Path = Path(self.cfg.raw_dataset_path)
        self.features = self.cfg.features
        self.target = self.cfg.target

        if self.raw_dataset_path.exists():
            shutil.rmtree(self.raw_dataset_path)
        os.makedirs(self.raw_dataset_path, exist_ok=True)

    def download_dataset(self):
        # download dataset from the HF
        # save to local dir (data/raw)
        try:
            log.info(f"Loading dataset from the repo: {self.dataset_repo}")
            dataset = load_dataset(path=self.dataset_repo, name=self.dataset_name)
            log.info(f"Saving dataset to load storage: {self.raw_dataset_path}")
            dataset.save_to_disk(dataset_dict_path=self.raw_dataset_path)
        except Exception as e:
            error: str = (
                f"Error while downloading {self.dataset_repo} dataset: {e!s}"
            )
            log.error(error)
            raise DataProcessingException(e) from e

    def load_dataset(self):
        # load from the load dir
        # return the dataset
        try:
            log.info(f"Loading dataset from the storage: {self.raw_dataset_path}")
            return load_from_disk(dataset_path=self.raw_dataset_path)
        except Exception as e:
            error: str = (
                f"Error while loading dataset from {self.raw_dataset_path}: {e!s}"
            )
            log.error(error)
            raise DataProcessingException(e) from e

    def eval_dataset(self, dataset: DatasetDict):
        # evaluate the dataset
        try:
            if hasattr(dataset, "keys") and len(dataset.keys()) > 0:
                first_split = list(dataset.keys())[0]
                available_columns = dataset[first_split].features
            else:
                available_columns = dataset.column_names

            # missing features
            missing_features = [f for f in self.features if f not in available_columns]
            if missing_features:
                error = f"Missing expected features in dataset: {missing_features}"
                log.error(error)
                raise ValueError(error)

            # target
            if self.target not in available_columns:
                error = f"Target column '{self.target}' not found in dataset columns: {available_columns}"
                log.error(error)
                raise ValueError(error)

        except Exception as e:
            error: str = f"Error while evaluation dataset: {e!s}"
            log.error(error)
            raise DataProcessingException(error) from e

    def split_dataset(
        self, dataset: DatasetDict, tokenizer: AutoTokenizer
    ) -> tuple[Dataset, Dataset, Dataset]:
        # split dataset to train, test,
        def tokenization_fn(examples):
            return tokenizer(examples["text"], truncation=True)

        try:
            log.info(
                "Splitting and tokenizing dataset into train, validation and test datasets"
            )
            tokenized_train = dataset["train"].map(tokenization_fn, batched=True)
            tokenized_validation = dataset["validation"].map(
                tokenization_fn, batched=True
            )
            tokenized_test = dataset["test"].map(tokenization_fn, batched=True)
            return tokenized_train, tokenized_validation, tokenized_test
        except Exception as e:
            error = f"Error while splitting dataset: {e!s}"
            log.error(error)
            raise DataProcessingException(error) from e
