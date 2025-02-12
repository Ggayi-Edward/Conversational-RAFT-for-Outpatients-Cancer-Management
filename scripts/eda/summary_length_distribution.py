import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_file = Path("../../outputs/plots/summary_length_distribution.png")
# Ensure input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: Input file not found at {input_file}")

# Load dataset
df = pd.read_csv(input_file)

# Ensure 'Cleaned_Summary' column exists
if "Cleaned_Summary" not in df.columns:
    raise ValueError("Error: 'Cleaned_Summary' column not found in dataset")

# Drop missing or empty summaries
df = df.dropna(subset=["Cleaned_Summary"])
df = df[df["Cleaned_Summary"].str.strip() != ""]

if df.empty:
    raise ValueError("Error: No valid summaries found after preprocessing!")

# Compute summary lengths
df["Summary_Length"] = df["Cleaned_Summary"].apply(lambda x: len(x.split()))

# Compute statistics
mean_length = np.mean(df["Summary_Length"])
median_length = np.median(df["Summary_Length"])

# Plot the distribution of summary lengths
plt.figure(figsize=(10, 6))
plt.hist(df["Summary_Length"], bins="auto", color="skyblue", edgecolor="black", alpha=0.7)
plt.axvline(mean_length, color="red", linestyle="dashed", linewidth=1, label=f"Mean: {mean_length:.1f}")
plt.axvline(median_length, color="green", linestyle="dashed", linewidth=1, label=f"Median: {median_length:.1f}")

plt.xlabel("Summary Length (Number of Words)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot
plt.tight_layout()
plt.savefig(output_file, dpi=300)

print(f"✅ Summary length distribution plot saved successfully at:\n📁 {output_file}")

# Optionally, show the plot
plt.show()
