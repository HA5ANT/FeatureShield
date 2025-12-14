import pandas as pd  # DataFrame operations and CSV loading
import numpy as np  # Numerical operations on arrays and lists
import xgboost as xgb  # XGBoost classifier with GPU support
from sklearn.model_selection import train_test_split  # Utility to split data into train/test sets
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score  # Evaluation metrics
from sklearn.preprocessing import LabelEncoder  # Encode labels safely
import joblib  # Model persistence (save/load trained models)

# ------------------------------
# Configuration constants
# ------------------------------
DATA_PATH = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"
MODEL_PATH = r"E:\code\FeatureShield_Phase1_clean\models\xgboost_ember.json"
NROWS = 50000  # Increased to get better class distribution (or use None for full dataset)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 50  # Minimum samples required per class for valid training

# ------------------------------
# STEP 1: Load data with random sampling to avoid label collapse
# ------------------------------
print(f"Loading {DATA_PATH}")
print("=" * 80)

# Read the ENTIRE dataset first (or a large chunk) to enable random sampling
df_full = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Full dataset size: {len(df_full)} rows, {df_full.shape[1]} columns")
# Check full label distribution
print("\nFull dataset label distribution:")
print(df_full['label'].value_counts())

# ------------------------------
# STEP 2: Filter out unlabeled samples (-1) explicitly
# ------------------------------
print("\n" + "=" * 80)
print("FILTERING OUT UNLABELED SAMPLES (label == -1)")
print("=" * 80)

df_labeled = df_full[df_full['label'].isin([0, 1])].copy()
print(f"Labeled samples: {len(df_labeled)} (removed {len(df_full) - len(df_labeled)} unlabeled)")
print("\nLabeled dataset distribution:")
print(df_labeled['label'].value_counts())

# Verify we have both classes
if df_labeled['label'].nunique() < 2:
    raise ValueError("❌ FATAL: Dataset contains only one class after filtering unlabeled samples!")

# ------------------------------
# STEP 3: Random sampling with stratification to ensure balanced subset
# ------------------------------
if NROWS and NROWS < len(df_labeled):
    print(f"\n{'=' * 80}")
    print(f"STRATIFIED RANDOM SAMPLING: {NROWS} samples")
    print("=" * 80)
    
    # Calculate samples per class to maintain ratio
    class_counts = df_labeled['label'].value_counts()
    min_class_size = class_counts.min()
    
    if NROWS < len(class_counts) * MIN_SAMPLES_PER_CLASS:
        print(f"⚠️  WARNING: NROWS={NROWS} is too small. Adjusting to minimum safe size.")
        NROWS = len(class_counts) * MIN_SAMPLES_PER_CLASS
    
    # Stratified sampling
    df = df_labeled.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), NROWS // len(class_counts)), random_state=RANDOM_STATE)
    ).reset_index(drop=True)
else:
    df = df_labeled.copy()

print(f"\nFinal working dataset: {len(df)} rows")
print("Final label distribution:")
print(df['label'].value_counts())
print()

# Verify minimum samples per class
for label_val in df['label'].unique():
    count = (df['label'] == label_val).sum()
    if count < MIN_SAMPLES_PER_CLASS:
        raise ValueError(f"❌ FATAL: Class {label_val} has only {count} samples (minimum: {MIN_SAMPLES_PER_CLASS})")

# ------------------------------
# STEP 4: Drop non-informative columns
# ------------------------------
drop_cols = ["Unnamed: 0", "avclass"]
existing_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_drop)
print(f"Dropped {len(existing_drop)} metadata columns: {existing_drop}")

# ------------------------------
# STEP 5: Convert timestamp to numeric feature
# ------------------------------
if "appeared" in df.columns:
    df["appeared_year"] = pd.to_datetime(df["appeared"], errors="coerce").dt.year
    df = df.drop(columns=["appeared"])
    print("Converted 'appeared' timestamp to 'appeared_year'")

# ------------------------------
# STEP 6: Enhanced feature engineering for list-encoded columns
# ------------------------------
print("\n" + "=" * 80)
print("FEATURE ENGINEERING: Processing list-encoded columns")
print("=" * 80)

list_cols = ["histogram", "byteentropy", "paths", "urls", "registry", "datadirectories"]
existing_list_cols = [c for c in list_cols if c in df.columns]

for col in existing_list_cols:
    print(f"Processing: {col}")
    
    # Parse pipe-delimited strings into numeric arrays
    df[col + "_array"] = df[col].apply(
        lambda x: [
            float(i) if str(i).replace(".", "", 1).replace("-", "", 1).isdigit() else 0
            for i in str(x).split("|")
        ]
    )
    
    # Create distributional features (not just sum/mean)
    df[col + "_sum"] = df[col + "_array"].apply(np.sum)
    df[col + "_mean"] = df[col + "_array"].apply(lambda arr: np.mean(arr) if len(arr) > 0 else 0)
    df[col + "_std"] = df[col + "_array"].apply(lambda arr: np.std(arr) if len(arr) > 1 else 0)
    df[col + "_max"] = df[col + "_array"].apply(lambda arr: np.max(arr) if len(arr) > 0 else 0)
    df[col + "_min"] = df[col + "_array"].apply(lambda arr: np.min(arr) if len(arr) > 0 else 0)
    
    # Drop original and temporary array columns
    df = df.drop(columns=[col, col + "_array"])

print(f"✓ Created distributional features for {len(existing_list_cols)} list columns")

# ------------------------------
# STEP 7: Remove duplicate samples by sha256
# ------------------------------
if "sha256" in df.columns:
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["sha256"])
    duplicates = before_dedup - len(df)
    print(f"\n✓ Removed {duplicates} duplicate samples (by sha256)")

# ------------------------------
# STEP 8: Separate features and labels BEFORE dropping object columns
# ------------------------------
y_raw = df["label"].copy()

# ------------------------------
# STEP 9: Drop remaining non-numeric columns (except identifiers we want to keep)
# ------------------------------
obj_cols = df.select_dtypes(include="object").columns.tolist()
keep_ids = []  # Add ["sha256", "md5"] if you need them later
obj_drop = [c for c in obj_cols if c not in keep_ids]

if obj_drop:
    df = df.drop(columns=obj_drop)
    print(f"\n✓ Dropped {len(obj_drop)} non-numeric columns")

# ------------------------------
# STEP 10: Build final feature matrix
# ------------------------------
X = df.drop(columns=["label"]).copy()
y = df["label"].copy()

# Clean numeric features
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.astype(np.float32)

print(f"\n{'=' * 80}")
print(f"FINAL DATASET STATISTICS")
print("=" * 80)
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")
print(f"Label distribution:\n{y.value_counts()}")
print()

# ------------------------------
# STEP 11: Stratified train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,  # Critical: maintains class balance
)

print(f"{'=' * 80}")
print("TRAIN/TEST SPLIT")
print("=" * 80)
print(f"Train samples: {X_train.shape[0]}")
print(f"Train label distribution:\n{y_train.value_counts()}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Test label distribution:\n{y_test.value_counts()}")
print()

# Verify both classes exist in train and test
if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError("❌ FATAL: Train or test set contains only one class!")

# ------------------------------
# STEP 12: Compute class imbalance weight for binary classification
# ------------------------------
scale_pos_weight = None
unique_labels = np.unique(y_train)

if len(unique_labels) == 2:
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    if num_pos > 0:
        scale_pos_weight = float(num_neg) / float(num_pos)
        print(f"{'=' * 80}")
        print(f"CLASS IMBALANCE HANDLING")
        print("=" * 80)
        print(f"Negative samples (0): {num_neg}")
        print(f"Positive samples (1): {num_pos}")
        print(f"scale_pos_weight: {scale_pos_weight:.4f}")
        print()

# ------------------------------
# STEP 13: Train XGBoost model with proper GPU configuration
# ------------------------------
print(f"{'=' * 80}")
print("TRAINING XGBOOST MODEL")
print("=" * 80)

# GPU parameters
gpu_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "logloss",
    }
if scale_pos_weight is not None:
    gpu_params["scale_pos_weight"] = scale_pos_weight

# CPU fallback parameters
cpu_params = dict(gpu_params)
cpu_params["device"] = "cpu"

# Try GPU first, fallback to CPU
try:
    print("Attempting GPU training...")
    model = xgb.XGBClassifier(**gpu_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("✓ GPU training successful")
except Exception as e:
    print(f"⚠️  GPU training failed: {e}")
    print("Falling back to CPU training...")
    model = xgb.XGBClassifier(**cpu_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print("✓ CPU training successful")

# ------------------------------
# STEP 14: Comprehensive evaluation with safety checks
# ------------------------------
print(f"\n{'=' * 80}")
print("MODEL EVALUATION")
print("=" * 80)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
unique_labels = np.unique(y_test)  # Define for ROC-AUC check


# Training and test accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")

# Classification report
print(f"\n{'─' * 80}")
print("CLASSIFICATION REPORT (Test Set)")
print("─" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix with explicit labels
print(f"{'─' * 80}")
print("CONFUSION MATRIX")
print("─" * 80)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)
print(f"\nFormat: [[TN, FP],\n         [FN, TP]]")

# ROC-AUC score (only for binary classification)
if len(unique_labels) == 2 and y_pred_proba.shape[1] == 2:
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"\n{'─' * 80}")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print("─" * 80)
    except Exception as e:
        print(f"\n⚠️  Could not compute ROC-AUC: {e}")

# ------------------------------
# STEP 15: Save model
# ------------------------------
model.save_model(MODEL_PATH)
print(f"\n{'=' * 80}")
print(f"✓ Model saved to: {MODEL_PATH}")
print("=" * 80)