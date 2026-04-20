## 1. Work completed so far

### Dataset setup and checking
- I checked that the EWS dataset can be loaded correctly from the local project path.
- I verified the split sizes:
  - train: 142 images
  - val: 24 images
  - test: 24 images
  - total: 190 images
- I checked that the image and mask sizes match correctly.
- I also checked that the masks are compatible with binary segmentation.
- I updated `src/config.py` so the local dataset path and output directories work correctly on my side.

### Metadata and sanity check
- I generated the split metadata files under `metadata/`:
  - `train.csv`
  - `val.csv`
  - `test.csv`
  - `all_splits.csv`
- I ran sanity-check visualisations to confirm that the original image, ground-truth mask, and overlay line up correctly.

### Running the three main methods
I ran all three segmentation methods on the test set successfully:
1. **Classical Lab + KMeans**
2. **Random Forest with handcrafted features**
3. **U-Net with ResNet18 encoder**

### Clean test-set summary results
From the current `results/summary.csv`, the clean-condition results are:

| Method | Macro Precision | Macro Recall | Macro F1 | Macro IoU |
|---|---:|---:|---:|---:|
| classical_lab_kmeans | 0.2903 | 0.0988 | 0.1403 | 0.0881 |
| ml_random_forest | 0.9572 | 0.7998 | 0.8621 | 0.7675 |
| dl_unet_r18 | 0.9305 | 0.9410 | 0.9351 | 0.8816 |

I also checked the micro metrics using `tools/read_summary.py`.

### Tools and analysis scripts I added
I added and tested several support scripts under `tools/`:

- `read_summary.py`
  - reads `results/summary.csv`
  - prints macro and micro metrics clearly

- `plot_metric_comparison.py`
  - plots metric comparison figures
  - currently used for macro IoU, macro F1, and training time comparison

- `visualize_dataset_splits.py`
  - creates overview figures for train / val / test
  - each figure shows the original image, mask, and overlay

- `visualize_method_comparison.py`
  - creates side-by-side prediction comparisons for the same test image

- `visualize_best_worst_cases.py`
  - selects representative good and bad U-Net examples and compares all methods on them

### Model saving
I updated the training scripts so trained models can be reused later:
- Random Forest model:
  - `results/ml_random_forest/rf_segmenter.pkl`
- U-Net checkpoint:
  - `results/dl_unet_r18/unet_segmenter.pth`

This is useful because robustness experiments can reuse saved models instead of retraining from scratch.

### Robustness workflow
I also completed the first working version of the robustness visualisation workflow.

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
From the current results, the overall trend is already quite clear:
- **U-Net is the most robust method overall** across the tested perturbations.
- **Random Forest is the second strongest method**, but it is more sensitive than U-Net when image quality drops.
- **Classical Lab + KMeans is the weakest method**, especially under brightness/contrast changes and Gaussian noise.
- In the **best case**, U-Net remains very stable under all four perturbations.
- In the **median case**, U-Net is still strong, while RF drops more noticeably.
- In the **worst case**, all methods degrade, but U-Net still usually performs best.

These results are already enough to support the robustness part of the report's Results and Discussion sections.

---

## 2. Work not finished yet

### Robustness section is not final yet
The robustness visualisation is working, but I have not fully turned it into the final report subsection yet.

Still missing:
- one compact robustness summary table
- final selection of which robustness figures should go into the report
- optional extra perturbation(s), such as blur or occlusion
- short written interpretation connecting robustness trends to the method design

### Report writing is still incomplete
The main code and figures are in place, but the writing itself is still not finished, especially:
- Results section draft
- Discussion section draft
- final figure/table selection
- concise explanation of why U-Net performs better than RF, and why RF performs better than the classical baseline

### Video material is also not final yet
I have the main figures ready, but the slides, speaking script, and recording plan are still not fully organised.

---

## 3. Suggested next steps

### Priority 1: finalise the robustness subsection
- pick the strongest 2-4 robustness figures
- make one compact summary table
- write a short paragraph summarising the main trend
- decide whether to include best + median, median + worst, or one strong case + one failure case in the main report

### Priority 2: prepare report-ready figures and tables
Use the current outputs to prepare:
- one final quantitative results table
- one metric comparison figure
- one dataset overview figure
- one method comparison figure
- one best/worst-case figure set
- one compact robustness figure set

### Priority 3: write Results and Discussion
A practical order would be:
1. dataset and split setup
2. short summary of the three methods
3. clean test quantitative results
4. qualitative comparison figures
5. best/worst cases
6. robustness analysis
7. discussion of why U-Net > RF > classical

### Priority 4: prepare video material
The following material is already usable for slides:
- dataset split overview figures
- metric comparison charts
- method comparison images
- best/worst case figures
- robustness figures

---

## 4. Commands used so far

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

### Print summary metrics
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

## 5. Short personal contribution summary

Up to this point, I mainly completed the experiment-running and visualisation support work on my side. This includes dataset verification, metadata generation, sanity-check visualisation, running all three segmentation methods, summarising the clean test results, adding several analysis/visualisation scripts, saving the RF and U-Net models for reuse, and implementing the first full robustness visualisation workflow based on representative best/median/worst cases.

At this stage, the main remaining work for me is no longer the basic coding. The more important next step is turning the current outputs into final report-ready figures/tables, writing the Results section and the robustness-related discussion, and then using those finished materials to prepare slides and speaking notes for the final video.
