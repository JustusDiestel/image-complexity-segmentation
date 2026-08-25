import matplotlib.pyplot as plt
import numpy as np


def show_result(
    image: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    metric_names: list[str],
) -> None:

    n_metrics = features.shape[-1]

    fig, axes = plt.subplots(
        1,
        n_metrics + 2,
        figsize=(5 * (n_metrics + 2), 5),
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(n_metrics):
        axes[i + 1].imshow(features[:, :, i])
        axes[i + 1].set_title(metric_names[i])
        axes[i + 1].axis("off")

    axes[-1].imshow(mask)
    axes[-1].set_title("Complexity Segmentation")
    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()