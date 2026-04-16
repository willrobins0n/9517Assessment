# Qishan Phase 1 handover (Apr 8-10)

## Completed deliverables

- dataset verification script
- split CSV index builder
- shared image/mask loading utilities
- preprocessing export script
- sanity-check visualisation script
- README for teammates

## Files to show the group

- `scripts/verify_dataset.py`
- `scripts/build_index.py`
- `scripts/preprocess.py`
- `scripts/sanity_check.py`
- `src/data_utils.py`
- `README_data.md`

## What this contribution covers

This completes the **preprocessing scripts** responsibility allocated to Qishan in Phase 1.
It also prepares the team for:
- ML-based feature extraction and classifier experiments
- deep learning data loading
- clean train / val / test separation
- later evaluation and report reproducibility

## Quick team update message

I completed the Phase 1 preprocessing package for the EWS dataset. It includes dataset verification, train/val/test CSV index generation, shared image-mask loading utilities, NPZ export for downstream pipelines, and sanity-check visualisations. This should support both the ML-based and deep learning methods while preserving the official split to avoid data leakage.
