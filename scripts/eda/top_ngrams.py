import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path


# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_file = Path("../../outputs/reports/top_ngrams.csv")

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


# Function to extract top n-grams (bigrams/trigrams)
def get_top_ngrams(corpus, ngram_range=(2, 2), top_n=10):
    """Extracts top n-grams from the given text corpus."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english", max_features=5000)
    ngram_matrix = vectorizer.fit_transform(corpus)
    ngram_counts = ngram_matrix.sum(axis=0).A1
    ngram_freq = [(ngram, ngram_counts[idx]) for ngram, idx in vectorizer.vocabulary_.items()]
    ngram_freq = sorted(ngram_freq, key=lambda x: x[1], reverse=True)
    return ngram_freq[:top_n]


# Extract top 10 bigrams (2-grams) and trigrams (3-grams)
top_bigrams = get_top_ngrams(df["Cleaned_Summary"], ngram_range=(2, 2), top_n=10)
top_trigrams = get_top_ngrams(df["Cleaned_Summary"], ngram_range=(3, 3), top_n=10)

# Convert to DataFrame for structured output
bigram_df = pd.DataFrame(top_bigrams, columns=["Bigram", "Frequency"])
trigram_df = pd.DataFrame(top_trigrams, columns=["Trigram", "Frequency"])

# Save results to CSV file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("Top 10 Bigrams:\n")
    bigram_df.to_csv(f, index=False)
    f.write("\nTop 10 Trigrams:\n")
    trigram_df.to_csv(f, index=False)

print(f"✅ Top n-grams saved successfully at: {output_file}")

# Print results
print("\n🔹 Top 10 Bigrams:")
print(bigram_df)
print("\n🔹 Top 10 Trigrams:")
print(trigram_df)
