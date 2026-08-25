import numpy as np

from complexity_seg.metrics.base import ComplexityMetric
from complexity_seg.segmenters.base import ComplexitySegmenter


class ComplexityPipeline:

    def __init__(
        self,
        metrics: list[ComplexityMetric],
        segmenter: ComplexitySegmenter,
    ):
        self.metrics = metrics
        self.segmenter = segmenter

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        feature_maps = []

        for metric in self.metrics:
            feature_map = metric.compute(image)
            feature_maps.append(feature_map)

        features = np.stack(
            feature_maps,
            axis=-1,
        )

        return features

    def run(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = self.extract_features(image)
        mask = self.segmenter.segment(features)

        return features, mask