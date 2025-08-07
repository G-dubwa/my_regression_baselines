import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
INPUT_CSV = "/home/glasson/Documents/Scripsie/my-baseline-regression/my_regression_baselines/data/age_gender.csv"
OUTPUT_CSV = "faces_sampled_uniform.csv"
TARGET_COUNT = 5000
RANDOM_SEED = 42

# -------------------------
# LOAD AND PREP
# -------------------------
df = pd.read_csv(INPUT_CSV)
df['age'] = df['age'].astype(int)

# Group by age
age_groups = df.groupby('age')
unique_ages = sorted(df['age'].unique())
num_ages = len(unique_ages)

samples_per_age = TARGET_COUNT // num_ages
selected_rows = []

random.seed(RANDOM_SEED)

# -------------------------
# SAMPLE UNIFORMLY
# -------------------------
for age in unique_ages:
    group = age_groups.get_group(age)
    if len(group) >= samples_per_age:
        sampled = group.sample(samples_per_age, random_state=RANDOM_SEED)
    else:
        sampled = group.copy()
    selected_rows.append(sampled)

final_df = pd.concat(selected_rows).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

if len(final_df) > TARGET_COUNT:
    final_df = final_df.iloc[:TARGET_COUNT]

# -------------------------
# SAVE
# -------------------------
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved uniformly sampled dataset ({len(final_df)} faces) to: {OUTPUT_CSV}")

# -------------------------
# PLOT BEFORE AND AFTER
# -------------------------
plt.figure(figsize=(14, 5))

# Before
plt.subplot(1, 2, 1)
df['age'].hist(bins=range(df['age'].min(), df['age'].max() + 1), color='skyblue', edgecolor='black')
plt.title("Original Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

# After
plt.subplot(1, 2, 2)
final_df['age'].hist(bins=range(df['age'].min(), df['age'].max() + 1), color='salmon', edgecolor='black')
plt.title("Sampled Age Distribution (Max 5000 Total)")
plt.xlabel("Age")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
