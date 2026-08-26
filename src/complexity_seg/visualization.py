import matplotlib.pyplot as plt
import numpy as np


def show_result(
    image: np.ndarray,
    features: np.ndarray,
    mask: np.ndarray,
    metric_names: list[str],
) -> None:

    n_metrics = features.shape[-1]

    # 1. Originalbild einzeln
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 2. Jede Complexity-Metrik einzeln
    for i in range(n_metrics):
        plt.figure(figsize=(6, 6))

        plt.imshow(
            features[:, :, i],
        )

        plt.title(metric_names[i])
        plt.axis("off")
        plt.colorbar(
            label="Complexity value",
        )

        plt.tight_layout()
        plt.show()

    # 3. Segmentierungsmaske einzeln
    plt.figure(figsize=(6, 6))

    plt.imshow(
        mask,
        vmin=0,
        vmax=2,
    )

    plt.title("Complexity Segmentation")
    plt.axis("off")

    colorbar = plt.colorbar(
        ticks=[0, 1, 2],
    )

    colorbar.ax.set_yticklabels(
        [
            "Low",
            "Medium",
            "High",
        ]
    )

    plt.tight_layout()
    plt.show()

    # 4. Alles zusammen
    fig, axes = plt.subplots(
        1,
        n_metrics + 2,
        figsize=(5 * (n_metrics + 2), 5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i in range(n_metrics):
        axes[i + 1].imshow(
            features[:, :, i],
        )

        axes[i + 1].set_title(
            metric_names[i]
        )

        axes[i + 1].axis("off")

    axes[-1].imshow(
        mask,
        vmin=0,
        vmax=2,
    )

    axes[-1].set_title(
        "Complexity Segmentation"
    )

    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()