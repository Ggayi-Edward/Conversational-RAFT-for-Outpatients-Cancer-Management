import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Define file path
input_file = Path("../../data/corpus/cleaned_corpus.csv")

# Ensure the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: Input file not found at {input_file}")

# Load dataset
df = pd.read_csv(input_file)

# Validate column existence
if "Cleaned_Summary" not in df.columns:
    raise ValueError("Error: 'Cleaned_Summary' column not found in dataset")

# Drop missing values
df = df.dropna(subset=["Cleaned_Summary"])
# Calculate sentence lengths (number of words per summary)
sentence_lengths = df["Cleaned_Summary"].apply(lambda x: len(x.split()))

# Plot box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x=sentence_lengths, color="royalblue")


plt.xlabel("Number of Words per Summary", fontsize=13)
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save and show plot
boxplot_path = Path("../../outputs/plots/sentence_length_distribution.png")
plt.savefig(boxplot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Sentence length distribution plot saved successfully at: {boxplot_path}")
