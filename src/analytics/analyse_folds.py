#!/usr/bin/env python3
import re
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
LOG_PATH = "logs/validation_per_epoch.txt"  # your .txt log
OUTPUT_DIR = "logs/analysis"                # where to save csvs/plots

# models appear in this order in the log; each has a fixed number of lines
MODELS_IN_ORDER = ["squeeze", "mobile", "resnet"]

# 10 dev folds * 9 test folds * 32 epochs = 2880 lines per model
LINES_PER_MODEL = 2880
EPOCHS_PER_COMBO = 32  # used for some sanity checks & summaries

# =========================
# PARSE LOG
# =========================
line_re = re.compile(
    r"epoch:\s*(?P<epoch>\d+),\s*training loss:\s*(?P<tr_loss>[-\d.]+),\s*"
    r"dev fold:\s*(?P<dev_fold>\d+),\s*dev loss:\s*(?P<dev_loss>[-\d.]+),\s*"
    r"test fold:\s*(?P<test_fold>\d+),\s*test loss:\s*(?P<test_loss>[-\d.]+),\s*"
    r"dev MAE:\s*(?P<dev_mae>[-\d.]+),\s*test MAE:\s*(?P<test_mae>[-\d.]+),\s*"
    r"dev RMSE:\s*(?P<dev_rmse>[-\d.]+),\s*test RMSE:\s*(?P<test_rmse>[-\d.]+),\s*"
    r"dev R2:\s*(?P<dev_r2>[-\d.]+),\s*test R2:\s*(?P<test_r2>[-\d.]+)"
)

rows = []
with open(LOG_PATH, "r") as f:
    for i, line in enumerate(f):
        m = line_re.search(line.strip())
        if not m:
            continue
        d = m.groupdict()
        # casts
        d["epoch"] = int(d["epoch"])
        for k in ["dev_fold", "test_fold"]:
            d[k] = int(d[k])
        for k in ["tr_loss", "dev_loss", "test_loss", "dev_mae", "test_mae",
                  "dev_rmse", "test_rmse", "dev_r2", "test_r2"]:
            d[k] = float(d[k])

        # map to model by block
        model_idx = i // LINES_PER_MODEL
        d["model"] = MODELS_IN_ORDER[model_idx] if model_idx < len(MODELS_IN_ORDER) else f"model_{model_idx}"
        rows.append(d)

df = pd.DataFrame(rows)

# keep only known models (in case log has extra trailing lines)
df = df[df["model"].isin(MODELS_IN_ORDER)].copy()

# =========================
# AGGREGATE: mean ± std per epoch (across fold combos)
# =========================
epoch_stats = (
    df.groupby(["model", "epoch"], as_index=False)
      .agg(
          n=("dev_r2", "count"),
          dev_r2_mean=("dev_r2", "mean"),
          dev_r2_std=("dev_r2", "std"),
          test_r2_mean=("test_r2", "mean"),
          test_r2_std=("test_r2", "std"),
          dev_rmse_mean=("dev_rmse", "mean"),
          test_rmse_mean=("test_rmse", "mean"),
          tr_loss_mean=("tr_loss", "mean"),
      )
      .sort_values(["model", "epoch"])
)

# sanity: count combos per epoch (should be number_of_combos = lines_per_model / epochs_per_combo)
combos_per_model = int(LINES_PER_MODEL // EPOCHS_PER_COMBO)

# =========================
# SUMMARIES per model
# =========================
summaries = []
for model in MODELS_IN_ORDER:
    sub = epoch_stats[epoch_stats["model"] == model].sort_values("epoch")
    if sub.empty:
        continue

    # final epoch metrics (mean across combos)
    last_row = sub.iloc[-1]
    final_epoch = int(last_row["epoch"])
    final_dev_r2_mean = float(last_row["dev_r2_mean"])
    final_test_r2_mean = float(last_row["test_r2_mean"])

    # epoch with max dev R2 mean
    idx_max_dev = sub["dev_r2_mean"].idxmax()
    row_max_dev = sub.loc[idx_max_dev]
    max_dev_epoch = int(row_max_dev["epoch"])
    max_dev_r2_mean = float(row_max_dev["dev_r2_mean"])
    test_r2_at_max_dev = float(row_max_dev["test_r2_mean"])

    # (optional) epoch with max test R2 mean
    idx_max_test = sub["test_r2_mean"].idxmax()
    row_max_test = sub.loc[idx_max_test]
    max_test_epoch = int(row_max_test["epoch"])
    max_test_r2_mean = float(row_max_test["test_r2_mean"])

    # area under curve over epochs (trapezoid), reflects sustained performance
    auc_dev = float(np.trapz(sub["dev_r2_mean"].values, sub["epoch"].values))
    auc_test = float(np.trapz(sub["test_r2_mean"].values, sub["epoch"].values))

    summaries.append({
        "model": model,
        "combos_per_epoch": combos_per_model,
        "epochs_counted": sub["epoch"].nunique(),
        "final_epoch": final_epoch,
        "final_dev_r2_mean": round(final_dev_r2_mean, 6),
        "final_test_r2_mean": round(final_test_r2_mean, 6),
        "max_dev_epoch": max_dev_epoch,
        "max_dev_r2_mean": round(max_dev_r2_mean, 6),
        "test_r2_at_max_dev_epoch": round(test_r2_at_max_dev, 6),
        "max_test_epoch": max_test_epoch,
        "max_test_r2_mean": round(max_test_r2_mean, 6),
        "auc_dev_r2_mean": round(auc_dev, 6),
        "auc_test_r2_mean": round(auc_test, 6),
    })

summary_df = pd.DataFrame(summaries)

# =========================
# SAVE CSVs
# =========================
outdir = Path(OUTPUT_DIR)
outdir.mkdir(parents=True, exist_ok=True)

epoch_stats_path = outdir / "epoch_means_by_model.csv"
summary_path = outdir / "summary_by_model.csv"

epoch_stats.to_csv(epoch_stats_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("Saved CSVs:")
print(" -", epoch_stats_path)
print(" -", summary_path)

# =========================
# PLOTS: learning curves (mean ± std)
# =========================
def plot_learning_curves(df_epoch_stats, model, outdir):
    sub = df_epoch_stats[df_epoch_stats["model"] == model].sort_values("epoch")
    if sub.empty:
        return

    epochs = sub["epoch"].values

    # --- R2 plot (dev & test with std bands) ---
    plt.figure(figsize=(8, 5))
    # dev
    dev_mean = sub["dev_r2_mean"].values
    dev_std = sub["dev_r2_std"].values
    plt.plot(epochs, dev_mean, label="Dev R² (mean)")
    if np.isfinite(dev_std).any():
        plt.fill_between(epochs, dev_mean - dev_std, dev_mean + dev_std, alpha=0.2, label="Dev R² (±1 std)")
    # test
    test_mean = sub["test_r2_mean"].values
    test_std = sub["test_r2_std"].values
    plt.plot(epochs, test_mean, label="Test R² (mean)")
    if np.isfinite(test_std).any():
        plt.fill_between(epochs, test_mean - test_std, test_mean + test_std, alpha=0.2, label="Test R² (±1 std)")

    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title(f"Learning Curves (R²) — {model}")
    plt.legend()
    plt.tight_layout()
    out_path = outdir / f"learning_curves_r2_{model}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot:", out_path)

    # --- RMSE plot (optional) ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, sub["dev_rmse_mean"].values, label="Dev RMSE (mean)")
    plt.plot(epochs, sub["test_rmse_mean"].values, label="Test RMSE (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curves (RMSE) — {model}")
    plt.legend()
    plt.tight_layout()
    out_path = outdir / f"learning_curves_rmse_{model}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot:", out_path)

for m in MODELS_IN_ORDER:
    plot_learning_curves(epoch_stats, m, outdir)

# =========================
# PRINT SUMMARY
# =========================
if not summary_df.empty:
    print("\nSummary by model (means across fold combos per epoch):")
    print(summary_df.to_string(index=False))
else:
    print("\nNo summary computed (empty DataFrame). Check your LOG_PATH/CONFIG.")
