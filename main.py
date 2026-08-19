import argparse

import yaml
from src.train_full import main as train_full
from src.train_lora import main as train_lora
from src.train_qlora import main as train_qlora
# from utils.handle_logging import get_logger

# log = get_logger(__file__)

# from utils.handle_config import sys_config

def load_config(path):
    """read config from given yaml file"""
    with open(path) as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to configs/full_ft.yaml")
    return p.parse_args()

def main():
    # log.info("Hello from efficient-fine-tuning-comparison!")
    # print(sys_config.base_config.model.repo)
    # cfg = load_config(path= "./config/full_ft.yaml")
    # a = parse_args()    
    # print(a.config)
    # train_full()
    # train_lora()
    train_qlora()



if __name__ == "__main__":
    main()
