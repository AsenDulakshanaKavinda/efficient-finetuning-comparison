from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.exceptions import DataProcessingException
from utils import get_logger, sys_config

log = get_logger(__file__)

model_checkpoint = sys_config.base_config.model.repo
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenization_fn(examples):
    return tokenizer(examples["text"], truncation=True)


class DataProcessing:
    def __init__(self):
        self.dataset_repo: str = sys_config.base_config.dataset.repo
        self.dataset_config: str = sys_config.base_config.dataset.config
        self.raw_dataset_path: Path = "../data/raw"
        self.features = sys_config.base_config.dataset.features
        self.target = sys_config.base_config.dataset.target

    def download_dataset(self):
        # download dataset from the HF
        # save to local dir (data/raw)
        try:
            log.info(f"Loading dataset from the repo: {self.dataset_repo}")
            dataset = load_dataset(path=self.dataset_repo, name=self.dataset_config)
            log.info(f"Saving dataset to load storage: {self.raw_dataset_path}")
            dataset.save_to_disk(dataset_dict_path=self.raw_dataset_path)
        except Exception as e:
            error: str = (
                f"Error while downloading {self.dataset_repo} dataset: {str(e)}"
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
                f"Error while loading dataset from {self.raw_dataset_path}: {str(e)}"
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
            error: str = f"Error while evaluation dataset: {str(e)}"
            log.error(error)
            raise DataProcessingException(error) from e

    def preprocess_dataset(self):
        # preprocess the dataset

        # drop columns

        # encode
        ...

    def split_dataset(self, dataset: DatasetDict) -> tuple[Dataset, Dataset, Dataset]:
        # split dataset to train, test, eval
        try:
            log.info("Splitting dataset into train, validation and test datasets")
            train, validation, test = (
                dataset["train"],
                dataset["validation"],
                dataset["test"],
            )
            return train, validation, test
        except Exception as e:
            error = f"Error while splitting dataset: {str(e)}"
            log.error(error)
            raise DataProcessingException(error) from e

    def tokenize_dataset(self, train_dataset: Dataset, validation_dataset: Dataset):
        # tokenize the dataset
        try:
            log.info("Creating tokenized train and validation dataset")
            tokenized_train = train_dataset.map(tokenization_fn, batched=True)
            tokenized_validation = validation_dataset.map(tokenization_fn, batched=True)
            return tokenized_train, tokenized_validation
        except Exception as e:
            error = f"Error while tokenizing dataset: {str(e)}"
            log.error(error)
            raise DataProcessingException(error) from e
