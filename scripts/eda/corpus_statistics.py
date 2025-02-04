import os
import pandas as pd
from pathlib import Path


# Define file paths
file_path = Path("../../data/corpus/cleaned_corpus.csv")
output_file = Path("../../outputs/reports/corpus_statistics.txt")
# Ensure the dataset exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: File not found at {file_path}")

# Load the dataset
df = pd.read_csv(file_path)

# Check if dataset is empty
if df.empty:
    raise ValueError("Error: The dataset is empty. Please check the source file.")

# Capture corpus statistics
stats = []
stats.append(f"📌 Total Documents: {len(df):,}")  # Adds thousand separator
stats.append(f"📌 Columns in Data: {', '.join(df.columns)}")

# Include first 5 documents for preview
stats.append("\n📜 First 5 Documents:\n")
stats.append(df.head().to_string(index=False))

# Save statistics to file
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure directory exists
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(stats))

print(f"✅ Corpus statistics saved successfully at:\n📁 {output_file}")
