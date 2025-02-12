import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
# Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Summary"])

# Reduce dimensions using PCA
pca = PCA(n_components=50)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Further reduce to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(pca_result)

# Scatter plot of t-SNE results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)

# Save visualization
tsne_plot_path = "../../outputs/plots/tsne_visualization.png"
plt.savefig(tsne_plot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ t-SNE visualization saved at: {tsne_plot_path}")
