import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your UTKFace CSV file
csv_path = "/home/glasson/Documents/Scripsie/my-baseline-regression/my_regression_baselines/data/age_gender.csv"  # replace with actual path

# Load the CSV
df = pd.read_csv(csv_path)

# Pick an example row
row = df.iloc[2]
pixels_str = row['pixels']
pixels = np.array(pixels_str.split(), dtype=np.uint8).reshape(48, 48)
normalized = pixels.astype(np.float32) / 255.0

# Plot
plt.figure(figsize=(4, 4))
plt.imshow(normalized, cmap='gray')
plt.title(f"Age: {row['age']}")
plt.axis('off')
plt.tight_layout()
plt.show()

