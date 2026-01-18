from abc import ABC, abstractmethod
import numpy as np

class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates pose keypoints and bounding boxes from an input frame.

        Args:
            frame: Input image as a numpy array.

        Returns:
            tuple: A tuple containing:
                - keypoints: Array of shape (N, 17, 3) where last dim is (y, x, score).
                             Coordinates should be normalized [0, 1].
                - bounding_boxes: Array of shape (N, 5) where last dim is (ymin, xmin, ymax, xmax, score).
                             Coordinates should be normalized [0, 1].
        """
        pass
