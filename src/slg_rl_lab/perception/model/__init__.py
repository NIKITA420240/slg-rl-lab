__all__ = ["PerceptionModel", "YOLOPerceptionModel", "registry_model"]
from .base import PerceptionModel
from .yolo import YOLOPerceptionModel

registry_model = {"yolo": YOLOPerceptionModel}
