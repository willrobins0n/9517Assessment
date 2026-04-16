# Qishan preprocessing package for COMP9517 EWS project

This package is designed for the **Phase 1 (Apr 8-10)** responsibilities allocated to **Qishan: preprocessing scripts** in the group project plan. It helps the team verify the EWS dataset, build split indexes, export preprocessed samples, and generate sanity-check visualisations.

## What this package includes

- `scripts/verify_dataset.py`  
  Checks dataset structure, image-mask pairing, shape consistency, and whether masks are binary-compatible.

- `scripts/build_index.py`  
  Builds `train.csv`, `val.csv`, `test.csv`, and `all_splits.csv` from the official dataset split.

- `scripts/preprocess.py`  
  Exports `.npz` files with preprocessed image and binary mask arrays for downstream ML / DL pipelines.

- `scripts/sanity_check.py`  
  Saves example figures showing `image / mask / overlay` for quick inspection.

- `src/data_utils.py`  
  Shared helper functions for split discovery, file pairing, image loading, mask loading, and simple validation.

## Recommended workflow

### 1. Verify dataset structure

```bash
python scripts/verify_dataset.py /path/to/EWS-Dataset
```

This writes a report to `metadata/dataset_summary.txt`.

### 2. Build split CSV files

```bash
python scripts/build_index.py /path/to/EWS-Dataset
```

This writes CSV files to `metadata/`.

### 3. Create sanity-check visualisations

```bash
python scripts/sanity_check.py metadata/train.csv
```

This writes example figures to `examples/sanity_checks/`.

### 4. Export preprocessed samples if needed

```bash
python scripts/preprocess.py metadata/train.csv --output-dir processed/train_npz
```

## Notes for teammates

- The package **does not re-split the dataset**. It preserves the official train / validation / test separation.
- Binary masks are loaded as `0/1` arrays by default.
- Images are loaded as RGB and normalized to `[0, 1]` by default.
- `preprocess.py` can also export grayscale, HSV, or Lab images for the later ML-based pipeline.
- If HSV or Lab is needed, install OpenCV (`opencv-python`).

## Suggested repo integration

Recommended project layout:

```text
project/
├─ data/
├─ scripts/
├─ src/
├─ metadata/
├─ examples/
└─ README.md
```

You can either:
1. copy these files into the team repo directly, or  
2. keep this package as a preprocessing module and import from it.

## Deliverables Qishan can report to the team

- dataset verification completed
- split CSV indexes created
- shared loader utilities created
- preprocessing export script created
- sanity-check plots generated

This is enough to count as a concrete Phase 1 contribution and also sets up the next Phase 2 ML-based method work.
