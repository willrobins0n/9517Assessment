from pathlib import Path
import sys
import pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.evaluation.metrics import score_masks
from src.methods.classical import LabKMeans
from src.methods.ml_rf import RandomForestSegmenter
from src.methods.deep_unet import UNetSegmenter


# -------------------------
# paths
# -------------------------
TEST_CSV = Path("metadata/test.csv")
UNET_PER_IMAGE_CSV = Path("results/dl_unet_r18_per_image.csv")

RF_MODEL_PATH = Path("results/ml_random_forest/rf_segmenter.pkl")
UNET_MODEL_PATH = Path("results/dl_unet_r18/unet_segmenter.pth")

OUTPUT_DIR = Path("tools_outputs/robustness_cases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# basic image helpers
# -------------------------
def load_rgb_image(image_path):
    return np.array(Image.open(image_path).convert("RGB"))


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask >= 128).astype(np.uint8)


def to_float01(image_uint8):
    return image_uint8.astype(np.float32) / 255.0


# -------------------------
# corruption functions
# -------------------------
def adjust_brightness(image, factor=0.5):
    """
    factor < 1.0 -> darker
    factor > 1.0 -> brighter
    """
    out = image.astype(np.float32) * factor
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def adjust_contrast(image, factor=0.7):
    """
    factor < 1.0 -> lower contrast
    factor > 1.0 -> higher contrast
    """
    image_f = image.astype(np.float32)
    mean = image_f.mean(axis=(0, 1), keepdims=True)
    out = (image_f - mean) * factor + mean
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def add_gaussian_noise(image, sigma=20):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


# -------------------------
# model loading
# -------------------------
def load_rf_segmenter():
    with open(RF_MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, RandomForestSegmenter):
        return obj

    seg = RandomForestSegmenter()

    # if only sklearn classifier was saved
    if hasattr(obj, "predict"):
        seg.clf = obj
        return seg

    if isinstance(obj, dict):
        if "segmenter" in obj and isinstance(obj["segmenter"], RandomForestSegmenter):
            return obj["segmenter"]
        if "clf" in obj:
            seg.clf = obj["clf"]
            return seg

    raise ValueError("Could not load RF model correctly.")


def clean_state_dict_keys(state_dict):
    clean_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        clean_dict[key] = value
    return clean_dict


def load_unet_segmenter():
    # keep settings simple here
    segmenter = UNetSegmenter()

    checkpoint = torch.load(UNET_MODEL_PATH, map_location=segmenter.device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)

    model = segmenter._build_model().to(segmenter.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    segmenter.model = model
    return segmenter


def build_methods():
    methods = {
        "Classical": LabKMeans(),
        "Random Forest": load_rf_segmenter(),
        "U-Net": load_unet_segmenter(),
    }
    return methods


# -------------------------
# select representative images
# -------------------------
def select_representative_cases():
    """
    Select 3 images based on U-Net per-image IoU:
    - best
    - median
    - worst
    """
    per_image_df = pd.read_csv(UNET_PER_IMAGE_CSV)
    test_df = pd.read_csv(TEST_CSV)

    merged_df = per_image_df.merge(
        test_df[["image_name", "image_path", "mask_path"]],
        on="image_name",
        how="inner"
    )

    if merged_df.empty:
        raise ValueError("Merged dataframe is empty. Check image_name matching.")

    sorted_df = merged_df.sort_values("iou", ascending=False).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    median_row = sorted_df.iloc[len(sorted_df) // 2]
    worst_row = sorted_df.iloc[-1]

    selected_df = pd.DataFrame([best_row, median_row, worst_row]).copy()

    print("\nSelected representative cases based on U-Net IoU:\n")
    print(
        selected_df[
            ["image_name", "iou", "f1"]
        ].to_string(index=False)
    )

    return selected_df


# -------------------------
# plotting
# -------------------------
def plot_case(
    image_name,
    original,
    corrupted,
    gt_mask,
    pred_results,
    corruption_name,
    case_label,
):
    fig, axes = plt.subplots(1, 6, figsize=(20, 4.8))

    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=11)

    axes[1].imshow(corrupted)
    axes[1].set_title(corruption_name.replace("_", " ").title(), fontsize=11)

    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth", fontsize=11)

    method_names = ["Classical", "Random Forest", "U-Net"]

    for i, method_name in enumerate(method_names, start=3):
        pred_mask = pred_results[method_name]["mask"]
        iou = pred_results[method_name]["iou"]
        f1 = pred_results[method_name]["f1"]

        axes[i].imshow(pred_mask, cmap="gray")
        axes[i].set_title(
            f"{method_name}\nIoU={iou:.3f} | F1={f1:.3f}",
            fontsize=10
        )

    for ax in axes:
        ax.axis("off")

    short_name = Path(image_name).stem
    fig.suptitle(
        f"{case_label} case | {short_name} | {corruption_name.replace('_', ' ')}",
        fontsize=13,
        y=0.98
    )

    plt.subplots_adjust(top=0.82, wspace=0.12)

    output_name = f"{case_label.lower()}_{Path(image_name).stem}_{corruption_name}.png"
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    np.random.seed(42)

    sample_df = select_representative_cases()
    methods = build_methods()

    corruption_dict = {
        "low_brightness": lambda img: adjust_brightness(img, factor=0.5),
        "low_contrast": lambda img: adjust_contrast(img, factor=0.6),
        "high_contrast": lambda img: adjust_contrast(img, factor=1.4),
        "gaussian_noise": lambda img: add_gaussian_noise(img, sigma=20),
    }

    # assign readable labels
    case_labels = ["Best", "Median", "Worst"]

    for case_label, (_, row) in zip(case_labels, sample_df.iterrows()):
        image_name = row["image_name"]
        original = load_rgb_image(row["image_path"])
        gt_mask = load_mask(row["mask_path"])

        print(f"\nProcessing {case_label} case: {image_name}")

        for corruption_name, corruption_fn in corruption_dict.items():
            corrupted = corruption_fn(original)
            corrupted_float = to_float01(corrupted)

            pred_results = {}

            for method_name, model in methods.items():
                pred_mask = model.predict(corrupted_float)
                pred_mask = np.asarray(pred_mask).squeeze().astype(np.uint8)

                scores = score_masks(pred_mask, gt_mask)

                pred_results[method_name] = {
                    "mask": pred_mask,
                    "iou": scores["iou"],
                    "f1": scores["f1"],
                }

            plot_case(
                image_name=image_name,
                original=original,
                corrupted=corrupted,
                gt_mask=gt_mask,
                pred_results=pred_results,
                corruption_name=corruption_name,
                case_label=case_label,
            )


if __name__ == "__main__":
    main()