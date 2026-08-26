import json
from pathlib import Path

import numpy as np
from PIL import Image


def export_sample(
    image: np.ndarray,
    mask: np.ndarray,
    output_dir: str | Path,
    sample_name: str,
    metadata: dict,
) -> None:
    output_dir = Path(output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    metadata_dir = output_dir / "metadata"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{sample_name}.png"
    mask_path = masks_dir / f"{sample_name}.png"
    preview_path = previews_dir / f"{sample_name}.png"
    metadata_path = metadata_dir / f"{sample_name}.json"

    _save_image(
        image,
        image_path,
    )

    _save_mask(
        mask,
        mask_path,
    )

    _save_mask_preview(
        mask,
        preview_path,
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


def _save_image(
    image: np.ndarray,
    path: Path,
) -> None:
    image = np.asarray(image)

    if np.issubdtype(
        image.dtype,
        np.floating,
    ):
        image = np.clip(
            image,
            0.0,
            1.0,
        )

        image = image * 255

    image = image.astype(
        np.uint8,
    )

    Image.fromarray(
        image,
    ).save(path)


def _save_mask(
    mask: np.ndarray,
    path: Path,
) -> None:
    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    Image.fromarray(
        mask,
    ).save(path)


def _save_mask_preview(
    mask: np.ndarray,
    path: Path,
) -> None:
    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    preview = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    preview[mask == 0] = [
        0,
        180,
        0,
    ]

    preview[mask == 1] = [
        255,
        165,
        0,
    ]

    preview[mask == 2] = [
        220,
        0,
        0,
    ]

    Image.fromarray(
        preview,
    ).save(path)