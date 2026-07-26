from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from .handle_logging import get_logger

log = get_logger(__file__)


class AdapterConfig(BaseModel): ...


class DatasetConfig(BaseModel):
    repo: str
    config: str
    features: list
    target: str


class ModelConfig(BaseModel):
    repo: str


class BaseConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig


class FullFtConfig(BaseModel): ...


class LoraConfig(BaseModel): ...


class QloraConfig(BaseModel): ...


class SysConfig(BaseModel):
    adapter_config: AdapterConfig
    base_config: BaseConfig
    full_ft_config: FullFtConfig
    lora_config: LoraConfig
    qlora_config: QloraConfig


config_files: dict = {
    "adapter_config": "adapter.yaml",
    "base_config": "base.yaml",
    "full_ft_config": "full_ft.yaml",
    "lora_config": "lora.yaml",
    "qlora_config": "qlora.yaml",
}


def load_config() -> SysConfig:
    config: dict = {}

    try:
        for k, config_path in config_files.items():
            log.info(
                f"Loading config from: {str(Path(__file__).resolve().parent.parent)}"
            )
            config_path = (
                Path(__file__).resolve().parent.parent / "config" / config_path
            )

            with open(config_path, "r") as file:
                raw_config = yaml.safe_load(file) or {}

            config[k] = raw_config

        sys_config = SysConfig(**config)

        return sys_config
    except ValidationError as ve:
        log.error(f"Error while loading config: {str(ve)}")
        raise ValidationError(ve) from ve
    except Exception as e:
        log.error(f"Error while loading config: {str(e)}")
        raise Exception(e) from e


sys_config = load_config()
