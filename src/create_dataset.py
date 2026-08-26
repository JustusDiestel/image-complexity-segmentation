from complexity_seg.batch import process_folder
from complexity_seg.metrics.delentropy import Delentropy
from complexity_seg.metrics.entropy import ShannonEntropy
from complexity_seg.metrics.mean_frequency import MeanFrequency
from complexity_seg.metrics.median_frequency import MedianFrequency
from complexity_seg.pipeline import ComplexityPipeline
from complexity_seg.segmenters.kmeans import KMeansSegmenter


pipeline = ComplexityPipeline(
    metrics=[
        ShannonEntropy(
            window_size=15,
            bins=32,
        ),
        Delentropy(
            window_size=15,
            bins=16,
        ),
        MeanFrequency(
            window_size=15,
        ),
        MedianFrequency(
            window_size=15,
        ),
    ],
    segmenter=KMeansSegmenter(
        n_clusters=3,
        random_state=42,
    ),
)


process_folder(
    input_dir="input_images",
    output_dir="examples/dataset",
    pipeline=pipeline,
)