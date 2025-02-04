import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# Define file paths
input_file = Path("../data/corpus/cleaned_corpus.csv")
output_file = Path("../data/corpus/tfidf_matrix.csv")


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
    max_features=5000,  # Limit vocab size for efficiency
    max_df=0.95,  # Ignore extremely common terms
    min_df=2  # Ignore rare terms
)

# Transform cleaned summaries into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Summary"])

# Convert TF-IDF matrix into a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Save to CSV
tfidf_df.to_csv(output_file, index=False)

# Display top TF-IDF words per document (optional)
num_top_words = 10
for idx, row in tfidf_df.iterrows():
    top_words = row.sort_values(ascending=False).head(num_top_words).index.tolist()
    print(f"🔹 Doc {idx + 1} Top Words: {', '.join(top_words)}")

print(f"✅ TF-IDF matrix successfully saved at: {output_file}")
