#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np

REQUIRED_COLS = ["Cough_ID", "Status", "Time_to_positivity"]
REMOVE_PATIENTS = {"CAGE0300", "CAGE0011"}  # blacklist set

def load_all_csvs(in_dir: str) -> pd.DataFrame:
    frames = []
    for fn in sorted(os.listdir(in_dir)):
        if fn.lower().endswith(".csv"):
            fp = os.path.join(in_dir, fn)
            df = pd.read_csv(fp)
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{fn} missing columns: {missing}")
            frames.append(df[REQUIRED_COLS].copy())
    if not frames:
        raise ValueError(f"No CSV files found in {in_dir}")
    return pd.concat(frames, ignore_index=True)

def make_random_folds(df: pd.DataFrame, n_folds: int, seed: int, group_by_patient: bool) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # extract Patient_ID = part before '/'
    df["Patient_ID"] = df["Cough_ID"].astype(str).str.split("/").str[0]

    # remove blacklisted patients
    df = df[~df["Patient_ID"].isin(REMOVE_PATIENTS)]

    # filter TTP == -1
    df = df[df["Time_to_positivity"] != -1].copy()

    folds = [list() for _ in range(n_folds)]

    if group_by_patient:
        patients = df["Patient_ID"].unique().tolist()
        rng.shuffle(patients)
        # assign patients round-robin to folds
        for i, pid in enumerate(patients):
            folds[i % n_folds].append(pid)
        # materialize rows per fold
        fold_dfs = []
        for i in range(n_folds):
            pids = set(folds[i])
            fold_dfs.append(df[df["Patient_ID"].isin(pids)][REQUIRED_COLS].reset_index(drop=True))
    else:
        # shuffle rows and split directly
        idx = np.arange(len(df))
        rng.shuffle(idx)
        splits = np.array_split(idx, n_folds)
        fold_dfs = [df.iloc[s][REQUIRED_COLS].reset_index(drop=True) for s in splits]

    return fold_dfs

def main():
    ap = argparse.ArgumentParser(description="Create completely random folds after filtering TTP == -1 and removing blacklisted patients.")
    ap.add_argument("--in_dir", default="data/cage/folds_with_ttp", help="Folder containing input CSVs")
    ap.add_argument("--out_dir", default="random_folds", help="Where to write the new folds")
    ap.add_argument("--n_folds", type=int, default=10, help="Number of folds to create")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no_group_by_patient", action="store_true",
                    help="If set, do NOT group coughs by patient (may cause leakage)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all_csvs(args.in_dir)
    fold_dfs = make_random_folds(
        df,
        n_folds=args.n_folds,
        seed=args.seed,
        group_by_patient=not args.no_group_by_patient
    )

    # write folds + quick summary
    summary_rows = []
    for i, fdf in enumerate(fold_dfs):
        out_path = os.path.join(args.out_dir, f"fold_{i}.csv")
        fdf.to_csv(out_path, index=False)

        # quick counts
        n_rows = len(fdf)
        n_patients = fdf["Cough_ID"].str.split("/").str[0].nunique()
        pos = int((fdf["Status"] == 1).sum())
        neg = int((fdf["Status"] == 0).sum())
        summary_rows.append({
            "fold": f"fold_{i}",
            "coughs": n_rows,
            "patients": n_patients,
            "coughs_pos": pos,
            "coughs_neg": neg
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nWrote {len(fold_dfs)} folds to: {args.out_dir}")

if __name__ == "__main__":
    main()
