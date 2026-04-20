from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# -------------------------
# paths
# -------------------------
PER_IMAGE_CSV = Path("results/dl_unet_r18_per_image.csv")
TEST_CSV = Path("metadata/test.csv")

CLASSICAL_DIR = Path("results/classical_lab_kmeans/predictions")
RF_DIR = Path("results/ml_random_forest/predictions")
UNET_DIR = Path("results/dl_unet_r18/predictions")

OUTPUT_DIR = Path("tools_outputs/best_worst_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rgb_image(image_path):
    return Image.open(image_path).convert("RGB")


def load_mask(mask_path):
    return Image.open(mask_path).convert("L")


def make_case_figure(row, image_path, mask_path, title_prefix):
    image_name = row["image_name"]

    original = load_rgb_image(image_path)
    gt_mask = load_mask(mask_path)

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

    iou_value = row["iou"]
    f1_value = row["f1"]
    fig.suptitle(
        f"{title_prefix}: {image_name} | U-Net IoU={iou_value:.4f}, F1={f1_value:.4f}",
        fontsize=11
    )
    fig.tight_layout()

    output_name = f"{title_prefix.lower()}_{Path(image_name).stem}.png"
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    per_image_df = pd.read_csv(PER_IMAGE_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # merge U-Net per-image metrics with test image/mask paths
    merged_df = per_image_df.merge(
        test_df[["image_name", "image_path", "mask_path"]],
        on="image_name",
        how="inner"
    )

    # sort by U-Net IoU
    sorted_df = merged_df.sort_values("iou", ascending=False)

    best_df = sorted_df.head(3)
    worst_df = sorted_df.tail(3)

    print("\nBest 3 cases (selected by U-Net IoU):\n")
    print(best_df[["image_name", "iou", "f1"]].to_string(index=False))

    print("\nWorst 3 cases (selected by U-Net IoU):\n")
    print(worst_df[["image_name", "iou", "f1"]].to_string(index=False))

    for _, row in best_df.iterrows():
        make_case_figure(
            row,
            row["image_path"],
            row["mask_path"],
            title_prefix="Best"
        )

    for _, row in worst_df.iterrows():
        make_case_figure(
            row,
            row["image_path"],
            row["mask_path"],
            title_prefix="Worst"
        )


if __name__ == "__main__":
    main()