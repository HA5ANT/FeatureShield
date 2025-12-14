# FeatureShield — Phase 1

A machine learning pipeline for training and evaluating XGBoost classifiers on EMBER-style malware detection features. The pipeline includes comprehensive feature engineering, GPU-accelerated training with CPU fallback, and generates presentation-ready evaluation visualizations and metrics.

---

## Features

- **Robust Data Processing**: Handles EMBER-style tabular features with automatic parsing of list-encoded columns
- **GPU Acceleration**: XGBoost training with CUDA support and automatic CPU fallback
- **Class Imbalance Handling**: Automatic computation of `scale_pos_weight` for binary classification
- **Comprehensive Evaluation**: Generates confusion matrix heatmaps, ROC curves, and metrics tables
- **Presentation-Ready Outputs**: High-resolution visualizations (300 DPI) suitable for reports

---

## Repository Structure

```
.
├── src/
│   └── traininggpu.py          # Main training and evaluation script
├── models/                      # Output directory for models and evaluation artifacts
│   ├── xgboost_ember.json       # Trained XGBoost model (output)
│   ├── confusion_matrix.png    # Confusion matrix visualization (output)
│   ├── roc_curve.png           # ROC curve visualization (output)
│   └── model_metrics.csv        # Metrics table (output)
├── data/raw/                    # Input data directory
│   └── train_features.csv       # EMBER-style feature CSV (input)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Environment Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, CPU fallback available)
- Windows/Linux/macOS

### Installation

**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### GPU Setup Notes

- On Windows, the `xgboost` pip wheel includes GPU support. Ensure you have an NVIDIA GPU with a recent driver installed.
- No extra CUDA package is typically required beyond the driver for the wheel to use CUDA.
- The script automatically falls back to CPU training if GPU is unavailable.

---

## Data Requirements

The training script expects a CSV file with the following structure:

### Required Columns

- **`label`**: Target column (binary: 0 = benign, 1 = malware)
  - Unlabeled samples (label = -1) are automatically filtered out

### Optional Columns

- **`appeared`**: Datetime-like string (converted to `appeared_year` feature)
- **`sha256`**: Sample identifier (used for deduplication)
- **List-encoded columns**: String columns with pipe-delimited values:
  - `histogram`, `byteentropy`, `paths`, `urls`, `registry`, `datadirectories`
  - These are automatically parsed and converted to distributional features (sum, mean, std, min, max)

### Columns Automatically Dropped

- `Unnamed: 0`, `avclass` (metadata columns)

---

## Usage

### Configuration

Edit the configuration constants at the top of `src/traininggpu.py`:

```python
DATA_PATH = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"
MODEL_PATH = r"E:\code\FeatureShield_Phase1_clean\models\xgboost_ember.json"
NROWS = 50000          # Number of samples to use (None for full dataset)
TEST_SIZE = 0.2       # Test set proportion
RANDOM_STATE = 42     # Random seed for reproducibility
MIN_SAMPLES_PER_CLASS = 50  # Minimum samples per class required
```

**Note**: For portability, consider using relative paths:
```python
DATA_PATH = "data/raw/train_features.csv"
MODEL_PATH = "models/xgboost_ember.json"
```

### Training

Run the training script:

```powershell
python src/traininggpu.py
```

### What the Script Does

1. **Data Loading**: Loads the full dataset and performs stratified random sampling to ensure balanced class distribution
2. **Filtering**: Removes unlabeled samples (label = -1) and verifies both classes are present
3. **Feature Engineering**:
   - Converts `appeared` timestamp to `appeared_year`
   - Parses list-encoded columns (histogram, byteentropy, etc.) into numeric arrays
   - Creates distributional features: sum, mean, std, min, max for each list column
   - Drops non-numeric columns (except identifiers)
4. **Deduplication**: Removes duplicate samples based on `sha256` hash
5. **Data Cleaning**: Replaces infinite values and NaN with 0, converts to float32
6. **Train/Test Split**: Stratified split maintaining class balance
7. **Class Imbalance**: Computes `scale_pos_weight` for binary classification
8. **Model Training**: 
   - Attempts GPU training first (CUDA)
   - Falls back to CPU if GPU unavailable
   - Uses XGBoost with optimized hyperparameters
9. **Evaluation**: Generates comprehensive metrics and visualizations
10. **Model Persistence**: Saves trained model in XGBoost native format

---

## Evaluation Outputs

After training, the script generates the following files in the `models/` directory:

### 1. Model Artifact
- **`xgboost_ember.json`**: Trained XGBoost model in native format

### 2. Visualizations

#### Confusion Matrix (`confusion_matrix.png`)
- Professional heatmap with annotations
- Shows True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP)
- Clear labels: "Benign (0)" and "Malware (1)"
- 300 DPI resolution for reports

#### ROC Curve (`roc_curve.png`)
- Receiver Operating Characteristic curve with AUC displayed
- Random classifier baseline for comparison
- Filled area under the curve
- AUC value prominently displayed
- 300 DPI resolution for reports

### 3. Metrics Table (`model_metrics.csv`)
- CSV file containing:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Suitable for importing into Excel, Word, or other reporting tools

### Console Output

The script also prints a summary to the console:
- Training and test accuracy
- Precision, Recall, F1-Score
- ROC-AUC (for binary classification)
- File paths for saved artifacts

---

## Model Configuration

The XGBoost model uses the following hyperparameters:

```python
{
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "tree_method": "hist",      # GPU: "hist", CPU: "hist"
    "device": "cuda",           # Falls back to "cpu" if unavailable
    "eval_metric": "logloss",
    "scale_pos_weight": <auto>  # Computed from class distribution
}
```

---

## Troubleshooting

### Memory Issues
- **Problem**: Out of memory (OOM) errors
- **Solution**: Reduce `NROWS` in the configuration, or process data in chunks

### GPU Not Detected
- **Problem**: Script falls back to CPU training
- **Solution**: 
  - Verify NVIDIA GPU driver is installed and up to date
  - Check GPU availability: `nvidia-smi` (Linux/Windows)
  - Ensure CUDA-compatible XGBoost is installed

### Path Errors
- **Problem**: FileNotFoundError when loading data
- **Solution**: Update `DATA_PATH` to use absolute or correct relative paths

### Single Class Error
- **Problem**: "FATAL: Dataset contains only one class"
- **Solution**: 
  - Check that your dataset has both benign (0) and malware (1) samples
  - Verify that unlabeled samples (-1) are being filtered correctly
  - Increase `NROWS` to get better class distribution

### High Accuracy (Potential Overfitting)
- **Problem**: Unusually high accuracy (>99%)
- **Solution**: 
  - Verify deduplication is working (check for `sha256` column)
  - Consider using group-aware splitting (e.g., `GroupShuffleSplit` by `sha256`)
  - Review feature engineering for potential data leakage

---

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `xgboost` - Gradient boosting classifier
- `scikit-learn` - Machine learning utilities and metrics
- `matplotlib` - Plotting library
- `seaborn` - Statistical visualizations
- `joblib` - Model persistence utilities

---

## License

For educational and defensive research purposes only. No warranties.

---

## Acknowledgments

This project uses EMBER-style features for malware detection. The feature engineering pipeline is designed to work with tabular feature representations commonly used in static malware analysis.
