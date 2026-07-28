import yaml

from utils.handle_logging import get_logger

# log = get_logger(__file__)

# from utils.handle_config import sys_config

def load_config(path):
    """read config from given yaml file"""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    # log.info("Hello from efficient-fine-tuning-comparison!")
    # print(sys_config.base_config.model.repo)
    cfg = load_config(path= "./config/full_ft.yaml")
    print(cfg)


if __name__ == "__main__":
    main()
