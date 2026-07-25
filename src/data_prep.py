
from pathlib import Path

from datasets import load_dataset, download
from src.exceptions import DataProcessingException

class DataProcessing:
    def __init__(self):
        self.dataset_repo: str = "dair-ai/emotion"
        self.dataset_config: str = "split"
        self.raw_dataset_path: Path = "../data/raw"


    def download_dataset(self):
        # download dataset from the HF
        # save to local dir (data/raw)
        try:
            dataset = load_dataset(path=self.dataset_repo, name=self.dataset_config)
            dataset.save_to_disk(dataset_dict_path=self.raw_dataset_path)
        except Exception as e:
            error:str = f"Error while downloading {self.dataset_repo} dataset: {str(e)}"
            raise DataProcessingException(error) from e
            
        

    def load_dataset(self):
        # load from the load dir
        # return the dataset
        ...

    def eval_dataset(self):
        # evaluate the dataset
        ...

    def preprocess_dataset(self):
        # preprocess the dataset
        ...

    def eval_preprocessed_dataset(self):
        # eval the preprocessed dataset
        ...

    def split_dataset(self):
        # split dataset to train, test, eval
        ...

    def tokenize_dataset(self):
        # tokenize the dataset
        ...
    

