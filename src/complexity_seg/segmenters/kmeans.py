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
            n_init=10,
        )

        labels = kmeans.fit_predict(scaled_features)

        ordered_labels = self._order_clusters(
            labels=labels,
            cluster_centers=kmeans.cluster_centers_,
        )

        return ordered_labels.reshape(height, width)

    def _order_clusters(
        self,
        labels: np.ndarray,
        cluster_centers: np.ndarray,
    ) -> np.ndarray:

        complexity_scores = cluster_centers.mean(axis=1)

        order = np.argsort(complexity_scores)

        mapping = {
            old_label: new_label
            for new_label, old_label in enumerate(order)
        }

        return np.array(
            [mapping[label] for label in labels],
            dtype=np.uint8,
        )