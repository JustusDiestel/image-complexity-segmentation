import numpy as np

from .base import ComplexityMetric


class MedianFrequency(ComplexityMetric):

    def __init__(self, window_size: int = 15):
        self.window_size = window_size

    @property
    def name(self) -> str:
        return "median_frequency"

    def compute(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image, dtype=np.float32)

        radius = self.window_size // 2

        padded = np.pad(
            image,
            radius,
            mode="reflect",
        )

        height, width = image.shape

        result = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        fy = np.fft.fftfreq(self.window_size)
        fx = np.fft.fftfreq(self.window_size)

        frequency_y, frequency_x = np.meshgrid(
            fy,
            fx,
            indexing="ij",
        )

        frequency_radius = np.sqrt(
            frequency_x**2 + frequency_y**2
        )

        flat_frequencies = frequency_radius.ravel()

        order = np.argsort(flat_frequencies)

        sorted_frequencies = flat_frequencies[order]

        for y in range(height):
            for x in range(width):

                window = padded[
                    y:y + self.window_size,
                    x:x + self.window_size,
                ]

                window = window - window.mean()

                spectrum = np.fft.fft2(window)

                power = np.abs(spectrum) ** 2

                flat_power = power.ravel()

                sorted_power = flat_power[order]

                cumulative_power = np.cumsum(sorted_power)

                total_power = cumulative_power[-1]

                if total_power > 0:
                    half_power = total_power / 2

                    index = np.searchsorted(
                        cumulative_power,
                        half_power,
                    )

                    result[y, x] = sorted_frequencies[index]

        return result