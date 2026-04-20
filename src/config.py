"""Project-wide paths.

Single source of truth for every filesystem path the project uses.
Every other script imports from here so wiring up the real dataset
means changing one line (DATASET_ROOT) instead of hunting through
scripts for hardcoded paths.
"""
from __future__ import annotations

from pathlib import Path

# Repo root derived from this file's location so paths stay correct
# regardless of where a script is invoked from (CLI, notebook, Colab).
# src/config.py -> src/ -> <repo root>.
REPO_ROOT = Path(__file__).resolve().parents[1]

# TODO: replace with the absolute path to your extracted EWS-Dataset
# folder once you're ready to wire up the real data. Leaving a sentinel
# path means anything that tries to actually read images before then
# will fail loudly instead of silently running on the wrong data.

DATASET_ROOT = REPO_ROOT / 'EWS-Dataset'

# Where scripts/build_index.py writes train.csv / val.csv / test.csv.
METADATA_DIR = REPO_ROOT / 'metadata'

# Where the evaluation harness writes per-image CSVs, summary.csv,
# and (optionally) saved prediction PNGs. Git-ignored.
RESULTS_DIR = REPO_ROOT / 'results'
