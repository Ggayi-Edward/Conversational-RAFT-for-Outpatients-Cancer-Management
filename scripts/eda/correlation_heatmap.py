import os
from textblob import TextBlob
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

# Perform sentiment analysis on each summary in the dataset



# Ensure the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: Input file not found at {input_file}")

# Load dataset
df = pd.read_csv(input_file)

df["Text_Length"] = df["Cleaned_Summary"].apply(lambda x: len(x.split()))


# Perform sentiment analysis on each summary in the dataset
sentiments = df["Cleaned_Summary"].apply(lambda x: TextBlob(x).sentiment.polarity)


# Validate column existence
if "Cleaned_Summary" not in df.columns:
    raise ValueError("Error: 'Cleaned_Summary' column not found in dataset")
# Assuming you have additional features like sentiment, text length, etc.
df["Sentiment"] = sentiments  # Adding sentiment column to the dataframe

# Calculate correlation matrix
correlation_matrix = df[["Text_Length", "Sentiment"]].corr()

# Plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.show()
