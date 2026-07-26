from utils.handle_logging import get_logger

log = get_logger(__file__)

from utils.handle_config import sys_config


def main():
    log.info("Hello from efficient-fine-tuning-comparison!")
    print(sys_config.base_config.model.repo)


if __name__ == "__main__":
    main()
