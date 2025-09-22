import os
import argparse
import pandas as pd

def summarize_folds(folds_dir: str) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(folds_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        fold_name = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(folds_dir, fname))

        # validate columns
        if "Cough_ID" not in df.columns or "Status" not in df.columns:
            raise ValueError(f"{fname} must have columns: Cough_ID, Status")

        # extract patient id from cough_id like "CAGE0002/1" -> "CAGE0002"
        df["Patient_ID"] = df["Cough_ID"].astype(str).str.split("/").str[0]

        # ----- patient-level counts -----
        patient_labels = df.groupby("Patient_ID", as_index=False)["Status"].max()
        n_patients_total = len(patient_labels)
        n_patients_pos = int((patient_labels["Status"] == 1).sum())
        n_patients_neg = int((patient_labels["Status"] == 0).sum())

        # ----- cough-level counts -----
        n_coughs_total = len(df)
        n_coughs_pos = int((df["Status"] == 1).sum())
        n_coughs_neg = int((df["Status"] == 0).sum())

        rows.append({
            "fold": fold_name,
            "patients_total": n_patients_total,
            "patients_positive": n_patients_pos,
            "patients_negative": n_patients_neg,
            "coughs_total": n_coughs_total,
            "coughs_positive": n_coughs_pos,
            "coughs_negative": n_coughs_neg,
        })

    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(description="Count TB-positive/negative patients and coughs per fold.")
    ap.add_argument("--folds_dir", default="data/cage/data_folds", help="path to folder with fold CSVs")
    ap.add_argument("--out_csv", default="fold_summary.csv", help="where to save the summary csv")
    args = ap.parse_args()

    summary = summarize_folds(args.folds_dir)
    print(summary.to_string(index=False))
    summary.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
