from pathlib import Path
import pandas as pd

summary_path = Path("results/summary.csv")

if not summary_path.exists():
    print("summary.csv not found.")
else:
    df = pd.read_csv(summary_path).round(4)

    print("\nMacro metrics:\n")
    print(
        df[
            [
                "method",
                "n_images",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "macro_iou",
            ]
        ].to_string(index=False)
    )

    print("\nMicro metrics:\n")
    print(
        df[
            [
                "method",
                "n_images",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "micro_iou",
            ]
        ].to_string(index=False)
    )