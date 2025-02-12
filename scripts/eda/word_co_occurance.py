import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import spacy
from spacy import displacy
from itertools import combinations
import networkx as nx
import umap
from sentence_transformers import SentenceTransformer

# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_plot_dir = Path("../../outputs/plots")
output_plot_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

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
word_freq = Counter(all_words)  # Frequency of words

# 1️⃣ **t-SNE Visualization of Word Embeddings (TF-IDF)**
vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Summary"])

# Reduce dimensions using PCA first
pca = PCA(n_components=50)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Further reduce to 2D using t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(pca_result)

# Plot t-SNE visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)

# Save t-SNE plot
tsne_plot_path = output_plot_dir / "tsne_visualization.png"
plt.savefig(tsne_plot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ t-SNE visualization saved at: {tsne_plot_path}")

# 2️⃣ **Hierarchical Clustering of Clinical Trial Summaries**
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Generate hierarchical clustering dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(cosine_sim, method='ward'))

plt.xlabel("Clinical Trial Summaries")
plt.ylabel("Distance")

# Save hierarchical clustering plot
hc_plot_path = output_plot_dir / "hierarchical_clustering.png"
plt.savefig(hc_plot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Hierarchical clustering plot saved at: {hc_plot_path}")

# 3️⃣ **BERT Sentence Embeddings with UMAP**
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["Cleaned_Summary"].tolist(), show_progress_bar=True)

# Reduce dimensionality using UMAP
umap_embeddings = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine").fit_transform(embeddings)

# Scatter plot of UMAP results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], alpha=0)
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.grid(True)

# Save UMAP visualization
bert_umap_path = output_plot_dir / "bert_umap.png"
plt.savefig(bert_umap_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ BERT UMAP visualization saved at: {bert_umap_path}")

# 4️⃣ **Word Co-Occurrence Network Graph**
top_words = [word for word, freq in word_freq.most_common(50)]

# Build co-occurrence network
co_occurrence = {}
for text in df["Cleaned_Summary"]:
    words = set(text.lower().split())
    words = [word for word in words if word in top_words]
    for pair in combinations(words, 2):
        co_occurrence[pair] = co_occurrence.get(pair, 0) + 1

# Create graph
G = nx.Graph()
for (word1, word2), weight in co_occurrence.items():
    G.add_edge(word1, word2, weight=weight)

# Draw network graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)


# Save network visualization
network_plot_path = output_plot_dir / "word_network.png"
plt.savefig(network_plot_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Word co-occurrence network saved at: {network_plot_path}")

# 5️⃣ **Heatmap of Term Co-Occurrence Matrix**
co_occurrence_matrix = pd.DataFrame(0, index=top_words, columns=top_words)
for (word1, word2), weight in co_occurrence.items():
    co_occurrence_matrix.loc[word1, word2] = weight
    co_occurrence_matrix.loc[word2, word1] = weight  # Ensure symmetry

# Plot heatmap of co-occurrence matrix
plt.figure(figsize=(12, 10))
sns.heatmap(co_occurrence_matrix, cmap="Blues", linewidths=0.5)


# Save heatmap visualization
heatmap_path = output_plot_dir / "word_heatmap.png"
plt.savefig(heatmap_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Co-occurrence heatmap saved at: {heatmap_path}")