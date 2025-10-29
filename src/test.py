import pandas as pd

# Path to your full CSV
csv_path = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"

# Chunk size (adjust based on your RAM, e.g., 100k rows)
chunksize = 100000

# Initialize counters
label_counts = {0: 0, 1: 0}

# Process CSV in chunks
for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=['label']):
    counts = chunk['label'].value_counts()
    for label, count in counts.items():
        if label in label_counts:
            label_counts[label] += count
        else:
            label_counts[label] = count

# Print results
print("Label distribution in the full dataset:")
for label, count in label_counts.items():
    print(f"Label {label}: {count}")
