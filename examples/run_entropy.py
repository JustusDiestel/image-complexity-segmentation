import matplotlib.pyplot as plt

from complexity_seg.metrics.entropy import ShannonEntropy
from complexity_seg.pipeline import ComplexityPipeline
from complexity_seg.segmenters.kmeans import KMeansSegmenter
from complexity_seg.visualization import show_result


image = plt.imread("test_image.png")

if image.ndim == 3:
    image = image.mean(axis=2)

pipeline = ComplexityPipeline(
    metrics=[
        ShannonEntropy(
            window_size=15,
            bins=32,
        )
    ],
    segmenter=KMeansSegmenter(
        n_clusters=3,
    ),
)

features, mask = pipeline.run(image)

show_result(
    image=image,
    features=features,
    mask=mask,
    metric_names=[
        metric.name
        for metric in pipeline.metrics
    ],
)