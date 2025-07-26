import pandas as pd
import matplotlib.pyplot as plt

file_path = "/Users/glassonwilliamosborne/Development/Phase 2/Scripsie/baseline_regression/data/cage/TimeToPositivityDataset.csv"
df = pd.read_csv(file_path)
total = len(df)
tb_positive = df[df["TB_status"]==1]
tb_total = len(tb_positive)
tb_positive = tb_positive[tb_positive["Time_to_positivity"]!=-1]
tb_valid = len(tb_positive)
ttp_values = tb_positive["Time_to_positivity"]

print("total: "+str(total)+" total tb: "+str(tb_total)+ " tb valid: "+str(tb_valid))
plt.figure(figsize=(8,5))
plt.hist(ttp_values, bins=20,edgecolor="black",alpha=0.7)
plt.title("Distribution of Time to Positivity (TB Positive Patients)")
plt.xlabel("Time to Positivity (days)")
plt.ylabel("Number of Patients")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()