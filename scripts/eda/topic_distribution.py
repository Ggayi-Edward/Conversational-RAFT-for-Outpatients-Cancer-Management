import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from pathlib import Path


# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_plot = Path("../../outputs/plots/topic_distribution.png")

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

# Prepare the feature matrix using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["Cleaned_Summary"])

# Apply Latent Dirichlet Allocation (LDA) for topic modeling
n_topics = 10  # Number of topics
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(X)

# Get the topic distribution for each document
topic_distribution = lda.transform(X)

# Determine the dominant topic for each document
dominant_topics = np.argmax(topic_distribution, axis=1)

# Plot the topic distribution
plt.figure(figsize=(12, 6))
sns.histplot(dominant_topics, bins=np.arange(n_topics + 1) - 0.5, kde=False, color="skyblue", edgecolor="black")
plt.title("Topic Distribution Across Clinical Trial Summaries", fontsize=14, fontweight="bold")
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Number of Documents", fontsize=12)
plt.xticks(range(n_topics))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

# Save and display the plot
plt.savefig(output_plot, dpi=300)
plt.show()

print(f"✅ Topic distribution plot saved successfully at: {output_plot}")

# Extract top words per topic
feature_names = vectorizer.get_feature_names_out()
top_words_per_topic = {}

for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]  # Top 10 words
    top_words_per_topic[f"Topic {topic_idx}"] = top_words

# Convert to DataFrame for easy inspection
topic_keywords_df = pd.DataFrame(top_words_per_topic)

# Save top words to a CSV file
output_csv = r"C:\Users\ECO11\Desktop\MINE\Conversational Retrieval Augmented Fine Tuning Outpatients Cancer Management\outputs\reports\topic_keywords.csv"
topic_keywords_df.to_csv(output_csv, index=False)

print(f"✅ Topic keywords saved successfully at: {output_csv}")

# Display extracted topics
print("\n🔹 Top Words per Topic:")
print(topic_keywords_df)
