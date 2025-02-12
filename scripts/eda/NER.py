import os
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import spacy
from collections import Counter
import seaborn as sns



# Load a spaCy model (use 'en_core_web_sm' or 'en_core_web_md' for better results)
nlp = spacy.load("en_core_web_sm")

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

# Extract named entities from the "Cleaned_Summary" column
entities = []
for text in df["Cleaned_Summary"]:
    doc = nlp(text)
    entities.extend([ent.label_ for ent in doc.ents])  # Get entity labels (types)

# Count entity occurrences
entity_counts = Counter(entities).most_common(10)  # Get top 10 entity types
entity_labels, entity_freqs = zip(*entity_counts)

# Plot entity distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=entity_freqs, y=entity_labels, palette="coolwarm")
plt.xlabel("Frequency", fontsize=13)
plt.ylabel("Entity Type", fontsize=13)
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Save and show plot
ner_plot_path = Path("../../outputs/plots/ner_distribution.png")
plt.savefig(ner_plot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Named entity recognition (NER) plot saved successfully at: {ner_plot_path}")
