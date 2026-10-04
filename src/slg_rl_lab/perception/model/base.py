from abc import ABC, abstractmethod
import numpy as np

class PerceptionModel(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray):
        """
        Predicts the output based on the input image.

        Args:
            image (np.ndarray): The input image for prediction.

        Returns:
            The predicted output.
        """
        raise NotImplementedError("The 'predict' method must be implemented in subclasses of PerceptionModel.")