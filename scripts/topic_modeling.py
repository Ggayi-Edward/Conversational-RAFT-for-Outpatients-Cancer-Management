import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import pyLDAvis
from pathlib import Path

# Download NLTK stopwords if needed
nltk.download("stopwords")

# Load SpaCy for lemmatization
nlp = spacy.load("en_core_web_sm")

# Define the path for the cleaned corpus
input_file = Path("../data/corpus/cleaned_corpus.csv")
# Load cleaned data
df = pd.read_csv(input_file)

# Function to clean and lemmatize text
def clean_and_lemmatize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stopwords.words("english")]
    return " ".join(tokens)

# Apply the preprocessing function to the 'Cleaned_Summary' column
df['Cleaned_Summary'] = df['Cleaned_Summary'].astype(str).apply(clean_and_lemmatize)

# Convert text to a document-term matrix
vectorizer = CountVectorizer(stop_words="english")
dtm = vectorizer.fit_transform(df["Cleaned_Summary"])

# Train LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42, max_iter=10, learning_method="batch")
lda.fit(dtm)

# Extract topics
topics = lda.components_
words = vectorizer.get_feature_names_out()

# Display top words per topic
for idx, topic in enumerate(topics):
    print(f"Topic {idx + 1}: {', '.join([words[i] for i in topic.argsort()[-10:]])}")

# Prepare pyLDAvis visualization (UPDATED)
vis = pyLDAvis.prepare(
    topic_term_dists=lda.components_ / lda.components_.sum(axis=1)[:, None],  # Normalize topic-word distributions
    doc_topic_dists=lda.transform(dtm),  # Get document-topic distributions
    doc_lengths=dtm.sum(axis=1).A1,  # Get document lengths
    vocab=vectorizer.get_feature_names_out(),  # Get feature names
    term_frequency=dtm.sum(axis=0).A1,  # Get term frequencies
)

# Save the visualization
pyLDAvis.save_html(vis, 'lda_visualization.html')

print("LDA model topics visualized and saved as 'lda_visualization.html'")
