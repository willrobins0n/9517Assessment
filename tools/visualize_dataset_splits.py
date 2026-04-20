from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


TRAIN_CSV = Path("metadata/train.csv")
VAL_CSV = Path("metadata/val.csv")
TEST_CSV = Path("metadata/test.csv")

OUTPUT_DIR = Path("tools_outputs/dataset_splits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


def load_mask(mask_path):
    return np.array(Image.open(mask_path).convert("L"))


def make_overlay(image, mask, alpha=0.35):
    overlay = image.copy().astype(np.float32)

    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255

    mask_binary = mask > 0
    overlay[mask_binary] = (
        (1 - alpha) * overlay[mask_binary] + alpha * red_mask[mask_binary]
    )

    return overlay.astype(np.uint8)


def visualize_split(csv_path, split_name, num_samples=4):
    df = pd.read_csv(csv_path)
    sample_df = df.head(num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image = load_rgb_image(row["image_path"])
        mask = load_mask(row["mask_path"])
        overlay = make_overlay(image, mask)

        axes[i][0].imshow(image)
        axes[i][0].set_title("Original")

        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title("Ground Truth")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay")

        for j in range(3):
            axes[i][j].axis("off")

    fig.suptitle(f"{split_name} split overview", fontsize=14)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{split_name}_overview.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    visualize_split(TRAIN_CSV, "train", num_samples=4)
    visualize_split(VAL_CSV, "val", num_samples=4)
    visualize_split(TEST_CSV, "test", num_samples=4)


if __name__ == "__main__":
    main()