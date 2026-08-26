import numpy as np
from scipy.ndimage import sobel

from .base import ComplexityMetric


class Delentropy(ComplexityMetric):

    def __init__(
        self,
        window_size: int = 15,
        bins: int = 16,
    ):
        self.window_size = window_size
        self.bins = bins

    @property
    def name(self) -> str:
        return "delentropy"

    def compute(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)

        gradient_x = sobel(image, axis=1, mode="reflect")
        gradient_y = sobel(image, axis=0, mode="reflect")

        radius = self.window_size // 2

        padded_x = np.pad(
            gradient_x,
            radius,
            mode="reflect",
        )

        padded_y = np.pad(
            gradient_y,
            radius,
            mode="reflect",
        )

        height, width = image.shape

        result = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        for y in range(height):
            for x in range(width):

                window_x = padded_x[
                    y:y + self.window_size,
                    x:x + self.window_size,
                ]

                window_y = padded_y[
                    y:y + self.window_size,
                    x:x + self.window_size,
                ]

                histogram, _, _ = np.histogram2d(
                    window_x.ravel(),
                    window_y.ravel(),
                    bins=self.bins,
                )

                probabilities = histogram / histogram.sum()
                probabilities = probabilities[probabilities > 0]

                result[y, x] = -np.sum(
                    probabilities
                    * np.log2(probabilities)
                )

        return result