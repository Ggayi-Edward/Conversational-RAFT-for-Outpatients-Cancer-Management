import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_plot = Path("../../outputs/plots/word_frequencies.png")

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

# Tokenize words and filter out non-alphabetic tokens
all_words = [word.lower() for text in df["Cleaned_Summary"] for word in text.split() if word.isalpha()]
word_freq = Counter(all_words)

# Get top 20 most frequent words
common_words = word_freq.most_common(20)
words, counts = zip(*common_words)

# Plot word frequencies
plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='royalblue')
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Words", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save and confirm plot
plt.savefig(output_plot, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Word frequency plot saved successfully at: {output_plot}")
