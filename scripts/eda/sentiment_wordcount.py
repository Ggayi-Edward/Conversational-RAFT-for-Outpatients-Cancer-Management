import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_plot_dir = Path("../../outputs/plots")
output_plot_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

# Ensure the input file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Error: Input file not found at {input_file}")

# Load dataset
df = pd.read_csv(input_file)

# Example Data: Sentiment vs. Word Count
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

# Plotting Sentiment vs. Word Count (Box Plot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='sentiment', y='word_count', palette='Set2')
plt.title('Sentiment vs. Word Count (Box Plot)')

# Saving the plot to the specified folder
output_path = output_plot_dir / "sentiment_vs_word_count.png"
plt.savefig(output_path)
plt.close()
