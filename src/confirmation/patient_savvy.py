import os
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
FOLDS_DIR = "data/cage/stratified_folds"  # Adjust this to your fold directory
FOLD_PREFIX = "fold_"  # prefix of the fold files
NUM_FOLDS = 10  # total number of folds

# -----------------------------
# Load all fold data
# -----------------------------
patient_to_folds = {}

for i in range(NUM_FOLDS):
    fold_path = os.path.join(FOLDS_DIR, f"{FOLD_PREFIX}{i}.csv")
    df = pd.read_csv(fold_path)

    # Extract Patient_ID from Cough_ID (assumes format "PatientID/whatever")
    df["Patient_ID"] = df["Cough_ID"].str.split("/").str[0]

    for pid in df["Patient_ID"]:
        if pid not in patient_to_folds:
            patient_to_folds[pid] = set()
        patient_to_folds[pid].add(i)

# -----------------------------
# Check for duplicates
# -----------------------------
violating_patients = {pid: folds for pid, folds in patient_to_folds.items() if len(folds) > 1}

if len(violating_patients) == 0:
    print("✅ All patients are uniquely assigned to a single fold.")
else:
    print(f"❌ {len(violating_patients)} patients occur in multiple folds.")
    print("\nExample violations:")
    for pid, folds in list(violating_patients.items())[:10]:  # Show first 10
        print(f" - Patient {pid} appears in folds: {sorted(folds)}")
