import pandas as pd
import os

# Paths
folds_path = "data/cage/stratified_folds"
ttp_file = "data/cage/TimeToPositivityDataset.csv"

# Load TTP data
ttp_data = pd.read_csv(ttp_file)

# Store summary rows
summary = []

# Loop through fold files
for fold_file in sorted(os.listdir(folds_path)):
    if fold_file.endswith(".csv"):
        # Load fold data
        fold_path = os.path.join(folds_path, fold_file)
        fold_df = pd.read_csv(fold_path)

        # Extract Patient_ID from Cough_ID
        fold_df["Patient_ID"] = fold_df["Cough_ID"].str.split("/").str[0]

        # Merge with TTP
        merged = fold_df.merge(
            ttp_data[["Patient_ID", "Time_to_positivity"]],
            on="Patient_ID",
            how="left"
        )


        status1_ttp_neg1_coughs = ((merged["Status"] == 1) & (merged["Time_to_positivity"] == -1)).sum()
        status1_ttp_valid_coughs = ((merged["Status"] == 1) & (merged["Time_to_positivity"] != -1)).sum()
        status0_ttp_neg1_coughs = ((merged["Status"] == 0) & (merged["Time_to_positivity"] == -1)).sum()

        status1_ttp_neg1_patients = merged[((merged["Status"] == 1) & (merged["Time_to_positivity"] == -1))]["Patient_ID"].nunique()
        status1_ttp_valid_patients = merged[((merged["Status"] == 1) & (merged["Time_to_positivity"] != -1))]["Patient_ID"].nunique()
        status0_ttp_neg1_patients = merged[((merged["Status"] == 0) & (merged["Time_to_positivity"] == -1))]["Patient_ID"].nunique()


        # Append row
        summary.append([
    fold_file,
    status1_ttp_neg1_coughs,
    status1_ttp_valid_coughs,
    status0_ttp_neg1_coughs,
    status1_ttp_neg1_patients,
    status1_ttp_valid_patients,
    status0_ttp_neg1_patients
])

# Create summary DataFrame
# Append row


# Create summary DataFrame
summary_df = pd.DataFrame(summary, columns=[
    "Fold",
    "Status1_TTP_-1_Coughs",
    "Status1_TTP_Valid_Coughs",
    "Status0_TTP_-1_Coughs",
    "Status1_TTP_-1_Patients",
    "Status1_TTP_Valid_Patients",
    "Status0_TTP_-1_Patients"
])

# Save to CSV
summary_df.to_csv("fold_summary.csv", index=False)

# Print
print(summary_df)
