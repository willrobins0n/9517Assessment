## 1. Work completed

### Clean test-set result summary and figure preparation
I used the current project outputs (especially `results/summary.csv`) to organise the clean-condition results into report-ready form.

The current clean test-set summary is:

| Method | Macro Precision | Macro Recall | Macro F1 | Macro IoU |
|---|---:|---:|---:|---:|
| classical_lab_kmeans | 0.2903 | 0.0988 | 0.1403 | 0.0881 |
| ml_random_forest | 0.9572 | 0.7998 | 0.8621 | 0.7675 |
| dl_unet_r18 | 0.9305 | 0.9410 | 0.9351 | 0.8816 |

I also used `tools/read_summary.py` to check the micro metrics so the clean-result ranking was consistent before writing the report section.

### Tools and analysis scripts I added
I added and tested several support scripts under `tools/` to turn the existing experiment outputs into report/video material:

- `read_summary.py`
  - reads `results/summary.csv`
  - prints macro and micro metrics clearly

- `plot_metric_comparison.py`
  - plots comparison figures
  - currently used for macro IoU, macro F1, and training time comparison

- `visualize_dataset_splits.py`
  - creates overview figures for train / val / test
  - each figure shows the original image, mask, and overlay

- `visualize_method_comparison.py`
  - creates side-by-side prediction comparisons for the same test image

- `visualize_best_worst_cases.py`
  - selects representative strong and weak U-Net examples and compares all methods on them

### Model reuse support for robustness experiments
I updated the workflow so the saved models could be reused directly in later analysis:
- Random Forest model:
  - `results/ml_random_forest/rf_segmenter.pkl`
- U-Net checkpoint:
  - `results/dl_unet_r18/unet_segmenter.pth`

This made it possible to run robustness visualisation without retraining models every time.

### Robustness workflow

The script `visualize_robustness_cases.py` now:
- loads the saved Random Forest and U-Net models
- uses the classical method together with RF and U-Net
- automatically selects three representative test cases based on U-Net per-image IoU:
  - **best**
  - **median**
  - **worst**
- applies four perturbations:
  - low brightness
  - low contrast
  - high contrast
  - Gaussian noise
- produces side-by-side figures showing:
  - original image
  - corrupted image
  - ground truth
  - classical prediction
  - random forest prediction
  - U-Net prediction
- writes IoU and F1 directly on each prediction panel

### Main observations from the current robustness figures
From the current robustness case-study figures, the overall trend is already clear:
- **U-Net is the most robust method overall** across the tested perturbations.
- **Random Forest is the second strongest method**, but it is more sensitive than U-Net when image quality drops.
- **Classical Lab + KMeans is the weakest method**, especially under brightness/contrast changes and Gaussian noise.
- In the **best case**, U-Net remains very stable under all four perturbations.
- In the **median case**, U-Net is still strong, while RF drops more noticeably.
- In the **worst case**, all methods degrade, but U-Net still usually performs best.

These results are already enough to support the robustness part of the report's Results and Discussion sections.

### Report writing
Based on the available figures and summary results, I added material to the shared google doc
- experimental setup
- clean test-set results summary
- qualitative examples
- robustness study
- some related discussion points around robustness and method trade-offs

---

## 2. Commands I used

### Read summary metrics
```bash
python3 tools/read_summary.py
```

### Create method comparison figures
```bash
python3 tools/visualize_method_comparison.py
```

### Create dataset split overview figures
```bash
python3 tools/visualize_dataset_splits.py
```

### Create metric comparison plots
```bash
python3 tools/plot_metric_comparison.py
```

### Create best/worst case comparison figures
```bash
python3 tools/visualize_best_worst_cases.py
```

### Create robustness comparison figures
```bash
python3 tools/visualize_robustness_cases.py
```

---

## 3. Short summary

Up to this point, my main work has been turning the current project outputs into analysis and presentation material. In particular, I focused on the robustness workflow, figure generation, comparison visualisations, and writing the report content that could be supported by the results already available. The part that is most clearly mine is the robustness analysis pipeline and the related Results-section material built from it.
