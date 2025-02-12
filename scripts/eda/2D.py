import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from pathlib import Path

# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_dir = Path("../../outputs/plots/")

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# File paths for saving images
heatmap_output_file = output_dir / "tfidf_heatmap.png"
pca_output_file = output_dir / "pca_projection.png"

# Ensure the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: Input file not found at {input_file}")

# Load cleaned data
df = pd.read_csv(input_file)

# Validate column existence
if "Cleaned_Summary" not in df.columns:
    raise ValueError("Error: 'Cleaned_Summary' column not found in dataset")

# Drop missing values
df = df.dropna(subset=["Cleaned_Summary"])

# Initialize TF-IDF vectorizer with optimizations
vectorizer = TfidfVectorizer(
    stop_words="english",  # Remove stopwords
    max_features=20,  # Limit vocab size for visualization (adjust based on needs)
    max_df=0.95,  # Ignore extremely common terms
    min_df=2  # Ignore rare terms
)

# Transform cleaned summaries into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Summary"])

# Convert TF-IDF matrix into a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save the TF-IDF DataFrame to CSV
tfidf_df.to_csv(output_dir / "tfidf_matrix.csv", index=False)

# Visualization 1: TF-IDF Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(tfidf_df, annot=False, cmap="YlGnBu", xticklabels=tfidf_df.columns, yticklabels=False)
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.title("TF-IDF Heatmap")
plt.savefig(heatmap_output_file)  # Save the heatmap as a PNG file
plt.close()  # Close the plot

# Visualization 2: 2D PCA Projection of the TF-IDF Matrix
pca = PCA(n_components=2)
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

plt.figure(figsize=(10, 6))
plt.scatter(tfidf_pca[:, 0], tfidf_pca[:, 1], c='teal', marker='o', alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of TF-IDF Matrix")
plt.savefig(pca_output_file)  # Save the PCA plot as a PNG file
plt.close()  # Close the plot

print(f"Visualizations saved to {output_dir}")
