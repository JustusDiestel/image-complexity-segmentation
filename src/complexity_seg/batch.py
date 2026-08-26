from pathlib import Path

import matplotlib.pyplot as plt

from complexity_seg.dataset_export import export_sample
from complexity_seg.pipeline import ComplexityPipeline


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


def process_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    pipeline: ComplexityPipeline,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    image_paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for image_path in image_paths:
        image = plt.imread(image_path)

        if image.ndim == 3:
            image = image[:, :, :3].mean(axis=2)

        features, mask = pipeline.run(image)

        metadata = {
            "source_file": image_path.name,
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
            "n_clusters": getattr(
                pipeline.segmenter,
                "n_clusters",
                None,
            ),
            "classes": {
                "0": "low",
                "1": "medium",
                "2": "high",
            },
        }

        export_sample(
            image=image,
            mask=mask,
            output_dir=output_dir,
            sample_name=image_path.stem,
            metadata=metadata,
        )