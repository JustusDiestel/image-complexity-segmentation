import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import ComplexitySegmenter


class KMeansSegmenter(ComplexitySegmenter):

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "kmeans"

    def segment(self, features: np.ndarray) -> np.ndarray:

        height, width, n_features = features.shape

        flat_features = features.reshape(
            -1,
            n_features,
        )

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(flat_features)

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )

        labels = kmeans.fit_predict(scaled_features)

        mask = labels.reshape(height, width)

        return mask