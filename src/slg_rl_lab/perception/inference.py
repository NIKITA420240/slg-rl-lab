from pathlib import Path
import json

import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from time import perf_counter
from loguru import logger

from .model import registry_model
from .visualization import draw_detections

@hydra.main(
    version_base=None,
    config_path="../configs/perception",
    config_name="inference",
)

def inference(cfg: DictConfig):
    """
    Run inference on the model with the given configuration.

    Args:
        cfg (DictConfig): Configuration for the inference process.
    """
    source = Path(cfg.source)
    output = Path(cfg.output)
    images_dir = output / "images"
    labels_dir = output / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    image_path = images_dir / f"{source.stem}_detections.png"
    json_path  = labels_dir / f"{source.stem}_detections.json"

    if source.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        raise ValueError(f"Unsupported file format: {source.suffix}")

    logger.info("Источник: {}", source)
    logger.info("Модель: {} | веса {}", cfg.model.name, cfg.model.model_path)

    model_class = registry_model[cfg.model.name]
    model       = model_class(cfg.model)

    with Image.open(source) as img:
        image_rgb = np.array(img.convert("RGB"))

    started = perf_counter()
    result  = model.predict(image_rgb)
    elapsed = perf_counter() - started

    count = len(result.boxes) if result.boxes is not None else 0
    logger.info("Найдено объектов: {} | инференс: {:.3f} с", count, elapsed)

    annotated = draw_detections(image_rgb, result)
    Image.fromarray(annotated).save(image_path)
    logger.info("Изображение сохранено: {}", image_path)

    inference_output = {
        "source": cfg.source,
        "detections": [
            {
                "class_id":   int(box.cls[0].item()),
                "class_name": result.names[int(box.cls[0].item())],
                "confidence": float(box.conf[0].item()),
                "xyxy":       list(map(int, box.xyxy[0].tolist())),
            }
            for box in result.boxes 
        ] if result.boxes is not None else []
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(inference_output, file, ensure_ascii=False, indent=4)
    logger.info("JSON сохранён: {}", json_path)

    return result