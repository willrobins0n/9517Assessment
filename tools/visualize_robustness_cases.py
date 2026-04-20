from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from src.methods.deep_unet import UNetSegmenter

# -------------------------
# paths
# -------------------------
TEST_CSV = Path("metadata/test.csv")
OUTPUT_DIR = Path("tools_outputs/robustness_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


def load_mask(mask_path):
    return np.array(Image.open(mask_path).convert("L"))


def darken_image(image, factor=0.5):
    """
    factor < 1.0 makes the image darker
    """
    dark = image.astype(np.float32) * factor
    dark = np.clip(dark, 0, 255)
    return dark.astype(np.uint8)


def plot_single_case(image_name, original, perturbed, gt_mask, pred_mask, perturb_name):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(perturbed)
    axes[1].set_title(perturb_name)

    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth")

    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{image_name} | robustness test: {perturb_name}", fontsize=11)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(image_name).stem}_{perturb_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(TEST_CSV)

    # first version: only first 3 test images
    sample_df = df.head(3)

    # create U-Net segmenter
    segmenter = UNetSegmenter()

    for _, row in sample_df.iterrows():
        image_name = row["image_name"]

        original = load_rgb_image(row["image_path"])
        gt_mask = load_mask(row["mask_path"])

        perturbed = darken_image(original, factor=0.5)

        pred_mask = segmenter.predict(perturbed)

        plot_single_case(
            image_name=image_name,
            original=original,
            perturbed=perturbed,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            perturb_name="low_brightness"
        )


if __name__ == "__main__":
    main()