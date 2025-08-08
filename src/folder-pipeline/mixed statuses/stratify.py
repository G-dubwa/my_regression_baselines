import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# CONFIG
# -------------------------
DATASET = "cage"
FOLDS = 10
ITERATIONS = 10000
OUTPUT_DIR = f"data/{DATASET}/stratified_folds"
DATA_FOLDS_PATH = f"data/{DATASET}/folds_with_ttp"
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD AND FILTER DATA
# -------------------------
all_files = [os.path.join(DATA_FOLDS_PATH, f) for f in os.listdir(DATA_FOLDS_PATH) if f.endswith(".csv")]
cough_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

if "Patient_ID" not in cough_df.columns:
    cough_df["Patient_ID"] = cough_df["Cough_ID"].str.split("/").str[0]

# Remove ONLY rows where Status == 1 AND TTP == -1
drop_mask = (cough_df["Status"] == 1) & (cough_df["Time_to_positivity"] == -1)
cough_df = cough_df.loc[~drop_mask].copy()

# For remaining rows, replace TTP == -1 with 43
cough_df.loc[cough_df["Time_to_positivity"] == -1, "Time_to_positivity"] = 43

# Build patient-level DataFrame (from filtered data)
patients = cough_df.groupby("Patient_ID").agg({
    "Status": "first"
}).reset_index()

# -------------------------
# SCORING FUNCTION
# -------------------------
def compute_metrics(patients_df, cough_df, folds, seed):
    np.random.seed(seed)
    shuffled = patients_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled["fold"] = np.tile(np.arange(folds), len(shuffled) // folds + 1)[:len(shuffled)]
    merged = cough_df.merge(shuffled[["Patient_ID", "fold"]], on="Patient_ID")

    # cough count balance
    fold_sizes = merged.groupby("fold")["Cough_ID"].count()
    cough_count_std = fold_sizes.std()

    # ttp balance (mean/std across folds)
    ttp_stats = merged.groupby("fold")["Time_to_positivity"].agg(["mean", "std"])
    ttp_mean_std = ttp_stats["mean"].std()
    ttp_std_std = ttp_stats["std"].std()

    return cough_count_std, ttp_mean_std, ttp_std_std, shuffled[["Patient_ID", "fold"]]

# -------------------------
# RUN MANY SPLITS
# -------------------------
seeds = np.arange(ITERATIONS)
with ThreadPoolExecutor() as executor:
    results = list(executor.map(
        lambda s: compute_metrics(patients_df=patients, cough_df=cough_df, folds=FOLDS, seed=s),
        seeds
    ))

# -------------------------
# SELECT BEST SPLIT
# -------------------------
cough_stds = np.array([r[0] for r in results])
mean_stds = np.array([r[1] for r in results])
std_stds = np.array([r[2] for r in results])

# Normalize each metric
cough_norm = cough_stds / (cough_stds.std() if cough_stds.std() > 0 else 1)
mean_norm = mean_stds / (mean_stds.std() if mean_stds.std() > 0 else 1)
std_norm = std_stds / (std_stds.std() if std_stds.std() > 0 else 1)

# Composite score (weights can be adjusted)
composite_scores = cough_norm + mean_norm + std_norm

best_idx = np.argmin(composite_scores)
best_split = results[best_idx][3]

print(f"✅ Best split selected (seed={best_idx})")

# -------------------------
# SAVE FOLDS
# -------------------------
final = cough_df.merge(best_split, on="Patient_ID")

for fold in range(FOLDS):
    fold_df = final[final["fold"] == fold][["Cough_ID", "Status", "Time_to_positivity"]]
    fold_df.to_csv(os.path.join(OUTPUT_DIR, f"fold_{fold}.csv"), index=False)

# -------------------------
# SUMMARY
# -------------------------
summary = final.groupby("fold")["Time_to_positivity"].agg(["count", "mean", "std"])
summary.to_csv(os.path.join(OUTPUT_DIR, "fold_distribution_summary.csv"))
print(summary)
