from abc import ABC, abstractmethod
import numpy as np


class ComplexityMetric(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric."""
        pass

    @abstractmethod
    def compute(self,image: np.ndarray):
        """Compute the metric for a given image."""
        pass