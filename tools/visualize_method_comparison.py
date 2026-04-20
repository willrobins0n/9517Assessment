from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -------------------------
# basic paths
# -------------------------
TEST_CSV = Path("metadata/test.csv")

CLASSICAL_DIR = Path("results/classical_lab_kmeans/predictions")
RF_DIR = Path("results/ml_random_forest/predictions")
UNET_DIR = Path("results/dl_unet_r18/predictions")

OUTPUT_DIR = Path("tools_outputs/method_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def load_mask(mask_path):
    # keep mask in grayscale so black/white is clear
    return Image.open(mask_path).convert("L")


def make_comparison_figure(row):
    image_name = Path(row["image_path"]).name

    original = load_image(row["image_path"])
    gt_mask = load_mask(row["mask_path"])

    classical_pred = load_mask(CLASSICAL_DIR / image_name)
    rf_pred = load_mask(RF_DIR / image_name)
    unet_pred = load_mask(UNET_DIR / image_name)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(gt_mask, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(classical_pred, cmap="gray")
    axes[2].set_title("Classical")

    axes[3].imshow(rf_pred, cmap="gray")
    axes[3].set_title("Random Forest")

    axes[4].imshow(unet_pred, cmap="gray")
    axes[4].set_title("U-Net")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_name, fontsize=12)
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(image_name).stem}_comparison.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(TEST_CSV)

    # first simple version: only visualize first 3 test images
    sample_df = df.head(3)

    for _, row in sample_df.iterrows():
        make_comparison_figure(row)


if __name__ == "__main__":
    main()