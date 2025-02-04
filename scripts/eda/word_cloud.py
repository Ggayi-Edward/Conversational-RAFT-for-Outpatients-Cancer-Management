import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path


# Define file paths

input_file = Path("../../data/corpus/cleaned_corpus.csv")
output_plot = Path("../../outputs/plots/word_cloud.png")
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

# Combine all cleaned summaries and filter out non-alphabetic words
all_words = " ".join(df["Cleaned_Summary"]).split()
filtered_text = " ".join([word.lower() for word in all_words if word.isalpha()])

# Generate word cloud
wordcloud = WordCloud(
    width=1000, 
    height=500, 
    background_color="white", 
    colormap="coolwarm",  # Improved color scheme
    max_words=200, 
    contour_color="black", 
    contour_width=1.5
).generate(filtered_text)

# Plot the word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("🔹 Word Cloud of Clinical Trial Summaries", fontsize=14, fontweight="bold")
plt.show()

# Save the word cloud image
wordcloud.to_file(output_image)

print(f"✅ Word cloud saved successfully at: {output_image}")
