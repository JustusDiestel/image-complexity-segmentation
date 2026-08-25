import numpy as np
from scipy.ndimage import generic_filter

from .base import ComplexityMetric


class ShannonEntropy(ComplexityMetric):

    def __init__(
        self,
        window_size: int = 15,
        bins: int = 32,
    ):
        self.window_size = window_size
        self.bins = bins

    @property
    def name(self) -> str:
        return "shannon_entropy"

    def compute(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)

        def local_entropy(values: np.ndarray) -> float:
            hist, _ = np.histogram(
                values,
                bins=self.bins,
                density=False,
            )

            probabilities = hist / hist.sum()
            probabilities = probabilities[probabilities > 0]
            return float(-np.sum(probabilities* np.log2(probabilities)))

        return generic_filter(
            image,
            local_entropy,
            size=self.window_size,
            mode="reflect",
        )