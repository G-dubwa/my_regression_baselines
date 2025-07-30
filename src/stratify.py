import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# CONFIG
DATASET = "cage"
FOLDS = 10
ITERATIONS = 10000
OUTPUT_DIR = f"data/{DATASET}/stratified_folds"
DATA_FOLDS_PATH = f"data/{DATASET}/data_folds"
TTP_FILE = f"data/{DATASET}/TimeToPositivityDataset.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load all cough-level data
all_files = [os.path.join(DATA_FOLDS_PATH, f) for f in os.listdir(DATA_FOLDS_PATH) if f.endswith(".csv")]
cough_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# Merge with TTP data
ttp_data = pd.read_csv(TTP_FILE)
cough_df["Patient_ID"] = cough_df["Cough_ID"].str.split("/").str[0]
merged = cough_df.merge(
    ttp_data[["Patient_ID", "Time_to_positivity"]],
    on="Patient_ID",
    how="left"
)

# Remove Type A: Status=1 & TTP=-1
filtered = merged[~((merged["Status"] == 1)&(merged["Time_to_positivity"] == -1))].copy()

# Build patient-level data
patients = filtered.groupby("Patient_ID").agg({
    "Status": "first"  # status is same for all coughs of a patient
}).reset_index()

# Global positive ratio
global_ratio = patients["Status"].mean()

def evaluate_split(seed, patients_df, folds=FOLDS):
    """
    Randomly assign patients to folds and compute score
    """
    np.random.seed(seed)
    shuffled = patients_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Assign folds
    shuffled["fold"] = np.tile(np.arange(folds), len(shuffled) // folds + 1)[:len(shuffled)]

    # Compute positive ratio per fold
    ratios = shuffled.groupby("fold")["Status"].mean()

    # Score = standard deviation of ratios
    score = ratios.std()

    return score, shuffled[["Patient_ID", "fold"]]

# Try multiple random splits in parallel
seeds = np.arange(ITERATIONS)
with ThreadPoolExecutor() as executor:
    results = list(executor.map(partial(evaluate_split, patients_df=patients), seeds))

# Select best split
best_score, best_split = min(results, key=lambda x: x[0])
print(f"Best split score (std dev of positive ratios): {best_score:.4f}")

# Merge best fold assignments back to cough-level data
final = filtered.merge(best_split, on="Patient_ID")

for fold in range(FOLDS):
    fold_df = final[final["fold"] == fold][["Cough_ID", "Status"]]
    fold_path = os.path.join(OUTPUT_DIR, f"fold_{fold}.csv")
    fold_df.to_csv(fold_path, index=False)


print(f"Stratified folds saved to {OUTPUT_DIR}")

# Print distribution summary
summary = final.groupby(["fold", "Status"])["Patient_ID"].nunique().unstack(fill_value=0)
summary["total_patients"] = summary.sum(axis=1)
summary["pos_ratio"] = summary[1] / summary["total_patients"]
print("\nFold distribution (patients):")
print(summary)