import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
filtered = merged[~((merged["Status"] == 1) & (merged["Time_to_positivity"] == -1))].copy()

# Build patient-level data (just to assign folds per patient)
patients = filtered.groupby("Patient_ID").agg({
    "Status": "first"  # status is same for all coughs of a patient
}).reset_index()

def compute_metrics(patients_df, cough_df, folds, seed):
    """
    Compute std of positive ratio, mean TTP, and std TTP for given random split
    """
    np.random.seed(seed)
    shuffled = patients_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["fold"] = np.tile(np.arange(folds), len(shuffled) // folds + 1)[:len(shuffled)]

    cough_with_folds = cough_df.merge(shuffled[["Patient_ID", "fold"]], on="Patient_ID")

    # Positive ratio std
    ratio_std = cough_with_folds.groupby("fold")["Status"].mean().std()

    # TTP metrics std (only valid TTPs)
    valid_ttp = cough_with_folds[cough_with_folds["Time_to_positivity"] != -1]
    ttp_stats = valid_ttp.groupby("fold")["Time_to_positivity"].agg(['mean', 'std'])
    mean_std = ttp_stats['mean'].std() if not ttp_stats['mean'].isna().all() else 0
    std_std = ttp_stats['std'].std() if not ttp_stats['std'].isna().all() else 0

    return ratio_std, mean_std, std_std, shuffled[["Patient_ID", "fold"]]

# Step 1: Compute metrics for all splits
seeds = np.arange(ITERATIONS)
with ThreadPoolExecutor() as executor:
    results = list(executor.map(
        lambda s: compute_metrics(patients_df=patients, cough_df=filtered, folds=FOLDS, seed=s),
        seeds
    ))

# Separate metrics
ratio_stds = np.array([r[0] for r in results])
mean_stds = np.array([r[1] for r in results])
std_stds = np.array([r[2] for r in results])

# Step 2: Normalize metrics (adaptive weighting)
ratio_norm = ratio_stds / (ratio_stds.std() if ratio_stds.std() > 0 else 1)
mean_norm = mean_stds / (mean_stds.std() if mean_stds.std() > 0 else 1)
std_norm = std_stds / (std_stds.std() if std_stds.std() > 0 else 1)

# Step 3: Composite score
composite_scores = ratio_norm + mean_norm + std_norm

# Step 4: Select best split
best_idx = np.argmin(composite_scores)
best_score = composite_scores[best_idx]
best_split = results[best_idx][3]

print(f"Best split composite score: {best_score:.4f}")
print(f"Raw metrics for best split -> "
      f"ratio_std: {ratio_stds[best_idx]:.4f}, "
      f"mean_std: {mean_stds[best_idx]:.4f}, "
      f"std_std: {std_stds[best_idx]:.4f}")

# Merge best fold assignments back to cough-level data
final = filtered.merge(best_split, on="Patient_ID")

# Save each fold
for fold in range(FOLDS):
    fold_df = final[final["fold"] == fold][["Cough_ID", "Status"]]
    fold_path = os.path.join(OUTPUT_DIR, f"fold_{fold}.csv")
    fold_df.to_csv(fold_path, index=False)

print(f"Stratified folds saved to {OUTPUT_DIR}")

# Print and save summary (cough-level positive ratio and TTP stats)
summary_pos = final.groupby(["fold", "Status"])["Cough_ID"].count().unstack(fill_value=0)
summary_pos["total_coughs"] = summary_pos.sum(axis=1)
summary_pos["pos_ratio"] = summary_pos[1] / summary_pos["total_coughs"]

valid_ttp_final = final[final["Time_to_positivity"] != -1]
summary_ttp = valid_ttp_final.groupby("fold")["Time_to_positivity"].agg(['mean', 'std'])

summary = summary_pos.join(summary_ttp)
summary.to_csv(os.path.join(OUTPUT_DIR, "fold_distribution_summary.csv"))
print("\nFold distribution (cough-level with TTP stats):")
print(summary)
