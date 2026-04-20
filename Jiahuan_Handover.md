## 1. What has been completed so far

### Environment and dataset setup
- Confirmed the EWS dataset directory is valid and readable.
- Verified the dataset splits and counts:
  - train: 142 images
  - val: 24 images
  - test: 24 images
  - total: 190 images
- Confirmed image and mask shapes are correct.
- Confirmed masks are binary-compatible.
- Updated `src/config.py` so project paths point to the local dataset and results directories correctly.

### Metadata generation and sanity checking
- Built split index files under `metadata/`:
  - `train.csv`
  - `val.csv`
  - `test.csv`
  - `all_splits.csv`
- Ran sanity-check visualisations and confirmed:
  - original image
  - ground-truth mask
  - overlay
  are correctly aligned.

### Baseline methods run successfully
Three segmentation methods have been run successfully on the test set:

1. **Classical Lab + KMeans**
2. **Random Forest with handcrafted features**
3. **U-Net with ResNet18 encoder**

### Current test-set results
Using the generated `results/summary.csv`, the current main results are:

| Method | Macro Precision | Macro Recall | Macro F1 | Macro IoU |
|---|---:|---:|---:|---:|
| classical_lab_kmeans | 0.2903 | 0.0988 | 0.1403 | 0.0881 |
| ml_random_forest | 0.9572 | 0.7998 | 0.8621 | 0.7675 |
| dl_unet_r18 | 0.9305 | 0.9410 | 0.9351 | 0.8816 |

Micro metrics were also checked and summarised using `tools/read_summary.py`.

### Visualisation / analysis tools added
The following tools were added under `tools/` and tested successfully:

- `read_summary.py`
  - Reads `results/summary.csv`
  - Prints macro and micro metrics in a clean format

- `plot_metric_comparison.py`
  - Plots summary metric comparison figures
  - Currently used for:
    - macro IoU comparison
    - macro F1 comparison
    - train time comparison

- `visualize_dataset_splits.py`
  - Creates overview images for train / val / test
  - Each figure shows:
    - original image
    - ground-truth mask
    - overlay

- `visualize_method_comparison.py`
  - Creates side-by-side comparisons for the same test image:
    - original
    - ground truth
    - classical prediction
    - random forest prediction
    - U-Net prediction

- `visualize_best_worst_cases.py`
  - Uses U-Net per-image IoU to select best 3 and worst 3 test cases
  - Produces multi-method comparison figures for each selected case

### Model saving added
Training scripts were updated so trained models are now saved for later reuse:

- Random Forest model saved to:
  - `results/ml_random_forest/rf_segmenter.pkl`

- U-Net checkpoint saved to:
  - `results/dl_unet_r18/unet_segmenter.pth`

This is important because robustness testing should ideally use already-trained models rather than retraining from scratch every time.

---

## 2. What has **not** been completed yet

### Robustness testing is not finished yet
The next major unfinished task is **robustness analysis**.

Planned robustness directions include testing method behaviour under perturbed inputs such as:
- low brightness
- low contrast
- blur
- noise
- possibly partial occlusion

This part has **not yet been completed** in a proper reusable way.

### Report writing is not finished yet
The code and visualisation groundwork is now strong, but the actual writing is still pending, especially:
- Results section draft
- Discussion section draft
- Final tables and figure selection

### Video preparation is not finished yet
There are now enough outputs and figures to support a strong video section, but the actual script/slides/recording are still pending.

---

## 3. Next steps

### Priority 1: finish robustness testing
Now that saved models exist, the best next step is:

1. load the saved RF and U-Net models
2. apply perturbations to selected test images
3. run inference on the perturbed inputs
4. compare:
   - original input vs perturbed input
   - original prediction vs perturbed prediction
   - cross-method robustness differences

First perturbation:
- low brightness

Then expand to:
- low contrast
- blur
- Gaussian noise

### Priority 2: prepare report-ready figures and tables
Use the existing outputs to assemble:
- one final results table
- one metric comparison figure set
- one dataset overview figure
- one method comparison figure
- one best/worst-case figure set

### Priority 3: write the Results and Discussion sections
A sensible order:
1. dataset + split setup
2. method summary
3. quantitative results
4. visual comparison of predictions
5. best/worst cases
6. robustness analysis
7. discussion of why U-Net > RF > classical

### Priority 4: prepare video material
Likely material already available for slides:
- dataset split overview figures
- metric comparison bar charts
- method comparison images
- best/worst cases

---

## 4. Console commands used so far

### Dataset verification
```bash
python3 scripts/verify_dataset.py EWS-Dataset
```

### Build metadata CSV files
```bash
python3 scripts/build_index.py EWS-Dataset
```

### Sanity-check split samples
```bash
python3 scripts/sanity_check.py metadata/train.csv
```

### Run classical method
```bash
python3 scripts/run_classical.py
```

### Run Random Forest method
```bash
python3 scripts/run_ml_rf.py
```

### Run U-Net method
```bash
python3 scripts/run_deep_unet.py
```

### Print summary metrics cleanly
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

---

## 5. Short contribution summary

I have now completed the main experiment-running and visualisation support work for the project. This includes dataset verification, metadata CSV generation, sanity-check visualisation, running all three segmentation methods, summarising the results, adding several analysis/visualisation tools, and updating the RF/U-Net scripts so trained models are saved for later reuse. The current next priority is robustness analysis using the saved models, followed by selecting final figures/tables for the report and preparing video materials.

