import os
import re
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
DATASET = "cage"
INPUT_DIR = f"data/{DATASET}/stratified_folds"
FOLDS = 10  # adjust if needed

# -------------------------
# LOAD ALL FOLDS
# -------------------------
fold_files = sorted(
    [f for f in os.listdir(INPUT_DIR) if re.match(r"fold_\d+\.csv$", f)],
    key=lambda x: int(re.findall(r"\d+", x)[0])
)
if not fold_files:
    raise FileNotFoundError(f"No fold_*.csv files found in {INPUT_DIR}")

dfs = []
for fold_num, file in enumerate(fold_files):
    df_f = pd.read_csv(os.path.join(INPUT_DIR, file))
    if "Patient_ID" not in df_f.columns:
        if "Cough_ID" not in df_f.columns:
            raise ValueError(f"{file} missing Cough_ID to derive Patient_ID")
        df_f["Patient_ID"] = df_f["Cough_ID"].astype(str).str.split("/").str[0]
    df_f["fold"] = fold_num
    dfs.append(df_f)

df_all = pd.concat(dfs, ignore_index=True)

# -------------------------
# CHECK 1: Unique Cough_ID
# -------------------------
dup_coughs = df_all[df_all.duplicated(subset=["Cough_ID"], keep=False)]
if dup_coughs.empty:
    print("✅ PASS: All Cough_IDs are unique across all folds")
else:
    print(f"❌ FAIL: Found {dup_coughs['Cough_ID'].nunique()} duplicate Cough_ID(s)")
    print(dup_coughs.sort_values("Cough_ID").head())

# -------------------------
# CHECK 2: Each patient only in one fold
# -------------------------
patient_folds = df_all.groupby("Patient_ID")["fold"].nunique()
leaky_patients = patient_folds[patient_folds > 1]
if leaky_patients.empty:
    print("✅ PASS: Each patient appears in only one fold")
else:
    print(f"❌ FAIL: {len(leaky_patients)} patients appear in multiple folds")
    print(leaky_patients.head())

# -------------------------
# FINAL VERDICT
# -------------------------
if dup_coughs.empty and leaky_patients.empty:
    print("\n🎯 All criteria satisfied.")
else:
    print("\n⚠ Please fix the above issues.")
