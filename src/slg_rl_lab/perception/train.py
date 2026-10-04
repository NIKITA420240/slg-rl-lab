import hydra
from loguru import logger
from omegaconf import DictConfig

from .model import registry_model

@hydra.main(
    version_base=None,
    config_path="../configs/perception",
    config_name="train",
)

def train(cfg: DictConfig):
    """
    """
    logger.info("Обучение модели: {}", cfg.model.model_path)

    model_class = registry_model[cfg.model.name]
    model       = model_class(cfg.model)

    return model.train(cfg)