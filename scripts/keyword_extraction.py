import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Define paths
input_file = Path("../data/corpus/cleaned_corpus.csv")
output_file = Path("../data/corpus/keywords_tfidf.csv")
def extract_keywords(input_path, output_path, max_features=50):
    """Extracts top keywords using TF-IDF from the cleaned corpus."""
    try:
        # ✅ Check if input file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' not found.")

        # ✅ Load cleaned corpus
        df = pd.read_csv(input_path)

        # ✅ Identify correct column containing summaries
        summary_col = None
        for col in df.columns:
            if "cleaned" in col.lower() and "summary" in col.lower():
                summary_col = col
                break
        
        if not summary_col:
            raise ValueError("No 'Cleaned_Summary' column found in the CSV file.")

        # ✅ Fill missing values with empty strings
        df[summary_col] = df[summary_col].fillna("")

        # ✅ TF-IDF setup
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),  # Capture unigrams and bigrams
            min_df=2  # Ignore rare words appearing in less than 2 documents
        )
        tfidf_matrix = vectorizer.fit_transform(df[summary_col])

        # ✅ Extract keywords
        keywords = vectorizer.get_feature_names_out()
        df_keywords = pd.DataFrame(tfidf_matrix.toarray(), columns=keywords)

        # ✅ Add Trial ID for reference if available
        if "Trial ID" in df.columns:
            df_keywords.insert(0, "Trial ID", df["Trial ID"])

        # ✅ Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ✅ Save keywords to CSV
        df_keywords.to_csv(output_path, index=False)
        logging.info(f"Keyword extraction completed. Output saved at: {output_path}")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
    except ValueError as e:
        logging.error(f"Value error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    extract_keywords(input_file, output_file)
