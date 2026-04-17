# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Single source of truth for every path used in the project.
# Target location in the repo: src/config.py
#
# Why have this file at all:
#   - Every other script (methods, harness, notebooks) imports DATASET_ROOT,
#     METADATA_DIR, RESULTS_DIR from here.
#   - When you wire up the real EWS dataset later, you change ONE line
#     (DATASET_ROOT below) and every script picks up the new path.
#   - No `../../../data` relative-path spaghetti anywhere else.
# =============================================================================

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# DATASET_ROOT — placeholder for now.
# -----------------------------------------------------------------------------
# TODO: when you want to wire up the real data, replace the string below with
# the absolute path to the extracted EWS-Dataset folder on your machine.
# Example after wiring:
#     DATASET_ROOT = Path('/Users/alex/data/EWS-Dataset')
# For now it stays as a sentinel so any script that accidentally tries to
# actually read data will fail loudly instead of silently using the wrong path.
DATASET_ROOT = Path('/PATH/TO/EWS-Dataset')

# -----------------------------------------------------------------------------
# REPO_ROOT — computed from this file's location.
# -----------------------------------------------------------------------------
# __file__ is the absolute path of this config.py. .resolve() turns it into a
# canonical absolute path. .parents[1] walks up one level:
#     src/config.py -> src/ -> <repo root>
# The reason to compute it instead of hardcoding: the path stays correct no
# matter where you run the script from (notebook CWD, CLI, Colab, etc.).
REPO_ROOT = Path(__file__).resolve().parents[1]

# Where scripts/build_index.py writes train.csv / val.csv / test.csv.
METADATA_DIR = REPO_ROOT / 'metadata'

# Where the evaluation harness writes per-image CSVs, summary.csv, and any
# saved prediction PNGs. You should add 'results/' to .gitignore — these are
# regenerated every run and shouldn't be in version control.
RESULTS_DIR = REPO_ROOT / 'results'
