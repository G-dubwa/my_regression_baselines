import pandas as pd
import os

# Paths
folds_path = "data/cage/stratified_folds"

# Store summary rows
summary = []

# Loop through fold files
for fold_file in sorted(os.listdir(folds_path)):
    if fold_file.endswith(".csv"):
        # Load fold_with_ttp file
        fold_path = os.path.join(folds_path, fold_file)
        df = pd.read_csv(fold_path)

        # Ensure expected columns exist
        assert "Status" in df.columns and "Time_to_positivity" in df.columns and "Cough_ID" in df.columns

        # Extract Patient_ID if needed
        if "Patient_ID" not in df.columns:
            df["Patient_ID"] = df["Cough_ID"].str.split("/").str[0]

        # Compute stats
        status1_ttp_neg1_coughs = ((df["Status"] == 1) & (df["Time_to_positivity"] == -1)).sum()
        status1_ttp_valid_coughs = ((df["Status"] == 1) & (df["Time_to_positivity"] != -1)).sum()
        status0_ttp_neg1_coughs = ((df["Status"] == 0) & (df["Time_to_positivity"] == -1)).sum()

        status1_ttp_neg1_patients = df[((df["Status"] == 1) & (df["Time_to_positivity"] == -1))]["Patient_ID"].nunique()
        status1_ttp_valid_patients = df[((df["Status"] == 1) & (df["Time_to_positivity"] != -1))]["Patient_ID"].nunique()
        status0_ttp_neg1_patients = df[((df["Status"] == 0) & (df["Time_to_positivity"] == -1))]["Patient_ID"].nunique()

        valid_ttp = df[df["Time_to_positivity"] != -1]["Time_to_positivity"]
        ttp_mean = valid_ttp.mean()
        ttp_std = valid_ttp.std()

        # Append row to summary
        summary.append([
            fold_file,
            status1_ttp_neg1_coughs,
            status1_ttp_valid_coughs,
            status0_ttp_neg1_coughs,
            status1_ttp_neg1_patients,
            status1_ttp_valid_patients,
            status0_ttp_neg1_patients,
            ttp_mean,
            ttp_std
        ])

# Create summary DataFrame
summary_df = pd.DataFrame(summary, columns=[
    "Fold",
    "Status1_TTP_-1_Coughs",
    "Status1_TTP_Valid_Coughs",
    "Status0_TTP_-1_Coughs",
    "Status1_TTP_-1_Patients",
    "Status1_TTP_Valid_Patients",
    "Status0_TTP_-1_Patients",
    "TTP_Mean",
    "TTP_Std"
])

# Save to CSV
summary_df.to_csv("fold_summary.csv", index=False)
print(summary_df)
