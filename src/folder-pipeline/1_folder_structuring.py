import os
import pandas as pd

# --------------------
# CONFIG
# --------------------
FOLDS_DIR = "/home/glasson/Documents/Scripsie/my-baseline-regression/my_regression_baselines/data/cage/data_folds"         # where your fold_X.csv files are
TTP_FILE = "/home/glasson/Documents/Scripsie/my-baseline-regression/my_regression_baselines/data/cage/TimeToPositivityDataset.csv"  # file with Patient_ID, Time_to_positivity
OUTPUT_DIR = "/home/glasson/Documents/Scripsie/my-baseline-regression/my_regression_baselines/data/cage/folds_with_ttp"          # new folder for enriched folds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# LOAD TTP DATA
# --------------------
ttp_df = pd.read_csv(TTP_FILE)
# Ensure Patient_ID is string
ttp_df["Patient_ID"] = ttp_df["Patient_ID"].astype(str)

# --------------------
# PROCESS EACH FOLD
# --------------------
for fold_file in sorted(os.listdir(FOLDS_DIR)):
    if fold_file.endswith(".csv"):
        fold_path = os.path.join(FOLDS_DIR, fold_file)
        fold_df = pd.read_csv(fold_path)

        # Extract Patient_ID from Cough_ID (before '/')
        fold_df["Patient_ID"] = fold_df["Cough_ID"].str.split("/").str[0].astype(str)

        # Merge with TTP data
        merged_df = fold_df.merge(
            ttp_df[["Patient_ID", "Time_to_positivity"]],
            on="Patient_ID",
            how="left"
        )

        # Drop Patient_ID column (not needed anymore)
        merged_df = merged_df.drop(columns=["Patient_ID"])

        # Save to new folder
        merged_path = os.path.join(OUTPUT_DIR, fold_file)
        merged_df.to_csv(merged_path, index=False)

        print(f"Saved enriched fold: {merged_path}")
