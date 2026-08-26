import matplotlib.pyplot as plt

from complexity_seg.metrics.entropy import ShannonEntropy
from complexity_seg.metrics.mean_frequency import MeanFrequency
from complexity_seg.metrics.delentropy import Delentropy
from complexity_seg.metrics.median_frequency import MedianFrequency
from complexity_seg.pipeline import ComplexityPipeline
from complexity_seg.segmenters.kmeans import KMeansSegmenter
from complexity_seg.visualization import show_result
from complexity_seg.dataset_export import export_sample

image = plt.imread("examples/test_images/fest.png")
global_window_size = 31

if image.ndim == 3:
    image = image.mean(axis=2)

pipeline = ComplexityPipeline(
    metrics=[
        ShannonEntropy(
            global_window_size,
            bins=16,
        ),
    ],
    segmenter=KMeansSegmenter(
        n_clusters=3,
        random_state=42,
    ),
)

features, mask = pipeline.run(image)

metadata = {
    "metrics": [
        {
            "name": metric.name,
            "window_size": getattr(
                metric,
                "window_size",
                None,
            ),
        }
        for metric in pipeline.metrics
    ],
    "segmenter": pipeline.segmenter.name,
    "n_clusters": pipeline.segmenter.n_clusters,
    "classes": {
        "0": "low",
        "1": "medium",
        "2": "high",
    },
}

export_sample(
    image=image,
    mask=mask,
    output_dir="examples/dataset",
    sample_name="fest",
    metadata=metadata,
)

metric_names = [
    metric.name
    for metric in pipeline.metrics
]

show_result(
    image=image,
    features=features,
    mask=mask,
    metric_names=[
        metric.name
        for metric in pipeline.metrics
    ],
)