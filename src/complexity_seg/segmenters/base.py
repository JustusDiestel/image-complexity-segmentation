from abc import ABC, abstractmethod
import numpy as np


class ComplexitySegmenter(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def segment(self, features: np.ndarray) -> np.ndarray:
        pass