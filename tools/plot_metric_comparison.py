from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


SUMMARY_CSV = Path("results/summary.csv")
OUTPUT_DIR = Path("tools_outputs/metric_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(SUMMARY_CSV)

    # add train times manually from experiment logs
    train_times = {
        "classical_lab_kmeans": 0.0,
        "ml_random_forest": 12.7,
        "dl_unet_r18": 130.8,
    }

    df["train_time_s"] = df["method"].map(train_times)

    # nicer labels for plots
    label_map = {
        "classical_lab_kmeans": "Classical",
        "ml_random_forest": "Random Forest",
        "dl_unet_r18": "U-Net",
    }

    df["plot_label"] = df["method"].map(label_map)

    # -------------------------
    # plot 1: macro IoU
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(df["plot_label"], df["macro_iou"])
    plt.title("Macro IoU Comparison")
    plt.ylabel("Macro IoU")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "macro_iou_comparison.png", dpi=200)
    plt.close()

    # -------------------------
    # plot 2: macro F1
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(df["plot_label"], df["macro_f1"])
    plt.title("Macro F1 Comparison")
    plt.ylabel("Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "macro_f1_comparison.png", dpi=200)
    plt.close()

    # -------------------------
    # plot 3: train time
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.bar(df["plot_label"], df["train_time_s"])
    plt.title("Training Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_time_comparison.png", dpi=200)
    plt.close()

    print("Saved metric plots to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()