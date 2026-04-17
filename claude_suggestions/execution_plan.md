# Execution plan — 33–36 tier, stretch to 37+

**Today:** 2026-04-17 (Fri)  
**Deadline:** 2026-04-24 18:00 Sydney (Fri) — 7 days.

Target: **Comprehensive tier (33–36)** via 3 meaningfully different
methods spanning at least 2 of the required categories (we actually
hit all 3 here). Stretch to **37+** on day 6–7 if ahead of schedule.

---

## The 3 methods

| # | Family | Method | Why it's there |
|---|--------|--------|----------------|
| 1 | Classical / unsupervised | K-means (or GMM) clustering in Lab colour space + morphological cleanup | No training, fast, interpretable, beats thresholding |
| 2 | ML + handcrafted features | Random Forest on per-pixel features: RGB + HSV + Lab + ExG vegetation index + small LBP/Gabor texture | Classical ML contrast, trains in seconds, strong baseline |
| 3 | Deep learning | U-Net (ResNet18 ImageNet-pretrained encoder) via `segmentation_models_pytorch` | DL contrast, trainable on CPU/Colab in a few minutes with 130 train images |

All three are evaluated with the **same** metrics harness on the
**same** test split: precision, recall, F1, IoU, plus training and
inference time.

---

## Day-by-day

### Day 1 — Fri 17 Apr  (today)
- Apply the two `data_utils.py` fixes from the audit.
- Decide repo layout for methods: add `src/methods/classical.py`,
  `src/methods/ml_rf.py`, `src/methods/deep_unet.py`, plus
  `src/evaluation/metrics.py`.
- Build a shared **evaluation harness**: loads a split CSV → runs a
  `predict(image) -> mask` callable → computes P/R/F1/IoU per image
  and aggregates → saves a results CSV + a summary dict. Every method
  plugs into this same harness.
- Placeholder for `DATASET_ROOT` — a single config value, no wiring
  yet.

### Day 2 — Sat 18 Apr
- **Method 1 (classical)**: Lab K-means (k=2 or 3, pick the "greenest"
  cluster), post-process with morphological opening + small-blob
  removal. Tune on val set.
- Quick notebook with side-by-side visualisations.

### Day 3 — Sun 19 Apr
- **Method 2 (RF)**: extract per-pixel features on the train split,
  train a RandomForestClassifier (sklearn), predict full masks on
  val/test. Keep a pixel-subsampling knob so training stays fast.

### Day 4 — Mon 20 Apr
- **Method 3 (U-Net)**: U-Net with pretrained ResNet18 encoder, BCE +
  Dice loss, light augmentation (flip/rotate/colour jitter), ~30
  epochs. Train on train, pick best by val IoU.
- Note training time & GPU/CPU used.

### Day 5 — Tue 21 Apr
- Run all 3 methods through the harness on the **test** split.
- Build the results table and pick 2–3 good examples and 2–3 failure
  examples per method for figures.
- Start the **report**: skeleton, intro, literature review, methods
  section.

### Day 6 — Wed 22 Apr
- Finish report draft: experimental setup, results, discussion,
  conclusion, references.
- Prepare slide deck.
- **If ahead**, start 37+ stretch (see below).

### Day 7 — Thu 23 Apr
- Record video (~10 min, ≤100 MB; coordinate the 5-member appearance
  requirement with teammates).
- Final report pass + check IEEE formatting, page limit, refs.
- Zip the code (≤25 MB; no dataset, no models, no result images).

### Buffer — Fri 24 Apr morning
- Final upload well before 18:00.

---

## Stretch to 37+ (only if day 5 finishes early)

Pick **one** of:

- **Robustness study**: perturb the test set (Gaussian noise, blur,
  brightness shift, random occlusion patches) at 2–3 severity levels,
  report how each method's IoU/F1 degrade. Extend U-Net with matching
  augmentations and show improvement. Clean half-day if harness is
  already in place.
- **Low-label ablation**: retrain the RF and U-Net on 25% / 50% /
  100% of training data; plot IoU vs. training-set size.

Either one, done carefully, is enough original-idea material.

---

## Guardrails

- Keep `DATASET_ROOT` as a placeholder until Alex says "wire it up".
- Never re-split the data. Tune on train+val only, touch test ONLY for
  final numbers.
- Save every experiment's predictions + metrics to disk — the report
  needs reproducible numbers.
- No new dependencies beyond `scikit-learn`, `scikit-image`,
  `torch`, `segmentation-models-pytorch`, `albumentations` (for U-Net
  aug). Add them to `requirements.txt` as you go.
