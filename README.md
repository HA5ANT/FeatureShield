# FeatureShield — Phase 1

Phase-1 prototype that trains and evaluates a Random Forest on EMBER-style tabular features (no raw binaries). Includes simple preprocessing, stratified splitting, evaluation, and model export.

---

## Status: Accuracy Issue Resolved

Earlier 1.0 / 1.0 train/test accuracy was traced to data issues. The pipeline now mitigates this by:

- Stratified train/test split to preserve label balance.
- Deduplication by `sha256` when present.
- Converting list-like feature strings into numeric aggregates and dropping residual object columns.

For even stronger isolation, you may adopt a group-aware split on `sha256` (see Recommendation below).

---

## Repository Structure

- `data/raw/train_features.csv` — EMBER-style features CSV.
- `data/raw/explore_dataset.ipynb` — optional EDA notebook.
- `src/training.py` — end-to-end training and evaluation script.
- `src/test.py` — utility to compute label distribution across the full dataset.
- `models/random_forest_ember.joblib` — trained model artifact (output).
- `requirements.txt` — pinned dependencies.

---

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

Train and evaluate:

```powershell
python src/training.py
```

Defaults in `src/training.py` (edit as needed):

```text
DATA_PATH  = E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv
MODEL_PATH = E:\code\FeatureShield_Phase1_clean\models\random_forest_ember.joblib
NROWS      = 200000
TEST_SIZE  = 0.2
RANDOM_STATE = 42
```

Tip: change to relative paths for portability, e.g. `data/raw/train_features.csv`.

Check label distribution on the full CSV:

```powershell
python src/test.py
```

---

## What the training script does

1. Loads up to `NROWS` from the CSV.
2. Drops clearly unused columns (e.g., `Unnamed: 0`, `avclass`).
3. Parses list-like string columns into numeric aggregates (sum/mean), then removes the raw strings.
4. Converts `appeared` to `appeared_year` and drops `appeared`.
5. Drops remaining object columns to keep only numeric features.
6. Deduplicates by `sha256` if available.
7. Performs stratified `train_test_split` and trains a `RandomForestClassifier`.
8. Prints a classification report, confusion matrix, and train/test accuracy.
9. Saves the model to `models/random_forest_ember.joblib`.

Recommendation: replace the split with `GroupShuffleSplit(groups=sha256)` for stronger duplicate isolation when identifiers are present.

---

## Troubleshooting

- Accuracy seems too high: confirm stratification and deduplication; consider group-aware split.
- Memory constraints: reduce `NROWS` or inspect via chunking (see `src/test.py`).
- Path issues: switch constants to relative paths.

---

## License and Usage

For educational and defensive research purposes only. No warranties.


