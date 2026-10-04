import cv2
import mss
import numpy as np
from time import sleep
from loguru import logger
import hydra
from omegaconf import DictConfig

from .model import registry_model
from .visualization import draw_detections
from .window import GameWindow

@hydra.main(
    version_base=None,
    config_path="../configs/perception",
    config_name="live",
)

def live_inference(cfg: DictConfig):
    """Capture screen frames, run detection, and display results."""
    window = GameWindow(cfg.capture.window_title)
    logger.info("Окно игры: {}", window.title)
    model_class = registry_model[cfg.model.name]
    model       = model_class(cfg.model)

    try:
        with mss.MSS() as capture:
            while True:
                region = window.get_region()
                if region is None:
                    if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                        break
                    sleep(0.1)
                    continue
                frame_bgra = np.asarray(capture.grab(region))
                image_rgb = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2RGB)
                result = model.predict(image_rgb, cfg)
                annotated = draw_detections(image_rgb, result)

                # OpenCV для отображения ожидает BGR.
                image_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow("Live detections", image_bgr)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # Q или Esc
                    break
    finally:
        cv2.destroyAllWindows()
