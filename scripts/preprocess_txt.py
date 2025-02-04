import os
import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path


# ✅ Download NLTK resources
nltk.download("stopwords")

# ✅ Load SpaCy model
nlp = spacy.load("en_core_web_sm")

input_file = Path("../data/corpus/clinical_trials_corpus.csv")  
output_file = Path("../data/corpus/cleaned_corpus.csv")

class ClinicalTrialPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with English stop words."""
        self.stop_words = set(stopwords.words("english"))

    def clean_text(self, text):
        """Cleans text by lowering case, removing special characters, and filtering stopwords."""
        if not isinstance(text, str):  # Ensure input is string
            return "N/A"
        
        text = text.lower()  # Convert text to lowercase
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters

        doc = nlp(text)
        tokens = [token.text for token in doc if token.text not in self.stop_words]

        return " ".join(tokens)

    def process_corpus(self, input_file, output_file):
        """Reads the input file, cleans the text, and saves the result to an output file."""
        try:
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"The input file '{input_file}' was not found.")

            df = pd.read_csv(input_file)

            # ✅ Debug: Print column names
            print("Columns in CSV:", df.columns.tolist())

            # ✅ Check for the correct column name
            summary_col = None
            for col in df.columns:
                if "summary" in col.lower():  # Case-insensitive match
                    summary_col = col
                    break

            if not summary_col:
                raise ValueError("The CSV file does not contain a column with 'Summary' in its name.")

            # ✅ Process and clean text
            df["Cleaned_Summary"] = df[summary_col].apply(self.clean_text)

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)

            print(f"Processed corpus saved to {output_file}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    preprocessor = ClinicalTrialPreprocessor()
    preprocessor.process_corpus(input_file, output_file)
