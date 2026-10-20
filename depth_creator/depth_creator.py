import numpy as np
from abc import ABC, abstractmethod


class DepthCreator(ABC):
    """Common interface of the ERP depth creator backends."""

    @abstractmethod
    def __init__(self) -> None:
        """Load the model."""

    @abstractmethod
    def run(self, img: np.ndarray) -> np.ndarray:
        """src ERP image → ERP depth map.

        Args:
            img: src ERP image, shape (H, W, 3), BGR uint8.

        Returns:
            ERP depth map (radius) in millimeters, shape (H, W), uint16.
        """
        raise NotImplementedError