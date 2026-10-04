from  .base import PerceptionModel
from ultralytics import YOLO
import numpy as np
from omegaconf import DictConfig

class YOLOPerceptionModel(PerceptionModel):
    def __init__(self, config: DictConfig):
        """
        Initializes the YOLOPerceptionModel with the specified model path.

        Args:
            model_path (str): The path to the YOLO model file.
        """
        self.config = config
        self.model = YOLO(config.model_path)

    def predict(self, image: np.ndarray):
        """
        Predicts the output based on the input image.

        Args:
            image (np.ndarray): The input image for prediction.

        Returns:
            The predicted output.
        """
        if (image.ndim != 3) or (image.shape[2] != 3):
            raise ValueError("Input image must be a 3D array with 3 channels (RGB).")
        if (image.dtype != np.uint8):
            raise ValueError("Input image must be of type uint8.")

        image_bgr = np.ascontiguousarray(image[..., ::-1])
        return self.model.predict(
            image_bgr,
            device=self.config.device,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            verbose=False,
        )[0]