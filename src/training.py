import pandas as pd  # DataFrame operations and CSV loading
import numpy as np  # Numerical operations on arrays and lists
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier implementation
from sklearn.model_selection import train_test_split  # Utility to split data into train/test sets
from sklearn.metrics import classification_report, confusion_matrix  # Evaluation metrics
import joblib  # Model persistence (save/load trained models)

# ------------------------------
# Configuration constants
# ------------------------------
# Absolute path to the training CSV containing features and labels
DATA_PATH = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"
# Absolute path where the trained Random Forest model will be saved
MODEL_PATH = r"E:\code\FeatureShield_Phase1_clean\models\random_forest_ember.joblib"
# Number of rows to read from the CSV for faster experimentation (use None to read all)
NROWS = 2000  # Adjust upward for better performance estimates if compute allows
# Proportion of samples to include in the test split (20% test, 80% train)
TEST_SIZE = 0.2
# Seed for reproducibility across runs (affects splitting and model initialization)
RANDOM_STATE = 42

# ------------------------------
# Load a subset (or full set) of rows from the CSV into a DataFrame
# ------------------------------
# Inform the user which file is being loaded and how many rows are requested
print(f"Loading {DATA_PATH} nrows= {NROWS}")
# Read the CSV; if NROWS is an int, read only that many rows for speed
df = pd.read_csv(DATA_PATH, nrows=NROWS)
# Report the initial shape of the raw dataset
print(f"Raw rows: {df.shape[0]} columns: {df.shape[1]}")

# ------------------------------
# Drop columns known to be non-informative for modeling
# ------------------------------
# "Unnamed: 0" is a typical index artifact from prior saves; "avclass" is metadata
drop_cols = ["Unnamed: 0", "avclass"]
# Remove these columns from the DataFrame
df = df.drop(columns=drop_cols)
# Log how many predefined useless columns were removed
print(f"Dropped {len(drop_cols)} useless columns")

# ------------------------------
# Convert the timestamp-like 'appeared' column into a numeric year feature
# ------------------------------
# Parse to datetime; coerce invalid values to NaT; then extract the year as an integer
df["appeared_year"] = pd.to_datetime(df["appeared"], errors="coerce").dt.year
# Drop the original textual datetime column to avoid leakage/redundancy
df = df.drop(columns=["appeared"])

# ------------------------------
# Convert string-encoded lists (pipe-delimited) into aggregate numeric features
# ------------------------------
# These columns may contain strings like "0.1|0.2|0.3" or mixed tokens; we summarize them
list_cols = ["histogram", "byteentropy", "paths", "urls", "registry", "datadirectories"]
for col in list_cols:
    # Split each string by '|' into tokens; convert numeric-looking tokens to float, else 0
    df[col + "_array"] = df[col].apply(
        lambda x: [
            float(i) if i.replace(".", "", 1).isdigit() else 0
            for i in str(x).split("|")
        ]
    )
    # Compute aggregate statistics from the parsed numeric list
    df[col + "_sum"] = df[col + "_array"].apply(np.sum)
    df[col + "_mean"] = df[col + "_array"].apply(np.mean)
    # Drop the original text column and the temporary array column to keep the feature space clean
    df = df.drop(columns=[col, col + "_array"])

# ------------------------------
# Drop any remaining non-numeric (object) columns to ensure the model sees only numbers
# ------------------------------
# Identify object-typed columns after prior transformations
obj_cols = df.select_dtypes(include="object").columns
# Remove them to avoid issues with scikit-learn estimators expecting numeric input
df = df.drop(columns=obj_cols)

# ------------------------------
# Remove exact duplicate samples identified by the 'sha256' identifier, if available
# ------------------------------
# Only proceed if the unique identifier column is present after prior drops
if "sha256" in df.columns:
    # Count duplicates to report how many will be removed
    duplicates = df.duplicated(subset=["sha256"]).sum()
    print(f"Dropped {duplicates} exact sha256 duplicates (if any).")
    # Keep the first occurrence and drop subsequent duplicates
    df = df.drop_duplicates(subset=["sha256"])

# ------------------------------
# Separate features matrix X and target vector y
# ------------------------------
# Features: all columns except the supervised learning target 'label'
X = df.drop(columns=["label"])
# Target: the binary/multiclass label column for classification
y = df["label"]

# ------------------------------
# Create stratified train/test splits to preserve class distribution
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,  # Maintain label proportions across splits
)

# Report the split sizes and final dimensionality of the feature space
print(f"Train rows: {X_train.shape[0]} Test rows: {X_test.shape[0]}")
print(f"Final feature count: {X_train.shape[1]}")

# ------------------------------
# Initialize and train a Random Forest classifier on the training data
# ------------------------------
# Use 100 trees for a balance of performance and speed; fix seed for reproducibility
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
# Fit the model to the training features and labels
rf.fit(X_train, y_train)

# ------------------------------
# Evaluate the trained model on the held-out test set
# ------------------------------
# Predict class labels for the test set
y_pred = rf.predict(X_test)
# Print a detailed precision/recall/F1 support report by class
print("\nClassification report (test set):\n")
print(classification_report(y_test, y_pred))

# Show the confusion matrix to visualize true/false positives/negatives
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
# Report accuracy on training data to gauge potential overfitting
print("Training accuracy:", rf.score(X_train, y_train))
# Report accuracy on test data as the primary generalization metric
print("Test accuracy   :", rf.score(X_test, y_test))

# ------------------------------
# Persist the trained model to disk for later inference/use
# ------------------------------
# Save using joblib which is efficient for scikit-learn estimators
joblib.dump(rf, MODEL_PATH)
# Confirm the output path so downstream steps know where to load from
print(f"Saved model to {MODEL_PATH}")
