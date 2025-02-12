import os
import pandas as pd
import spacy
from spacy import displacy
from pathlib import Path

# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_svg = Path("../../outputs/plots/dependency_tree.svg")

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

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Select a sample clinical trial summary for visualization
sample_text = df["Cleaned_Summary"].iloc[0]  # Take the first summary
doc = nlp(sample_text)

# Display dependency tree in Jupyter Notebook (if running in a notebook)
displacy.render(doc, style="dep", jupyter=True, options={"compact": True, "color": "royalblue", "bg": "#f8f9fa"})

# Save dependency tree as an SVG file
svg = displacy.render(doc, style="dep", options={"compact": True, "color": "royalblue", "bg": "#f8f9fa"})
output_svg.parent.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
with open(output_svg, "w", encoding="utf-8") as f:
    f.write(svg)

print(f"✅ Dependency tree visualization saved successfully at: {output_svg}")
