import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import spacy

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

# Load SpaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Initialize a directed graph
G = nx.DiGraph()

# Process summaries and extract entities with relationships
for text in df["Cleaned_Summary"][:77]:  # Limit to 100 texts for simplicity
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Add edges with entity labels as relationships
    for i in range(len(entities) - 1):
        G.add_edge(entities[i][0], entities[i + 1][0], label=entities[i + 1][1])  # Edge label as entity type

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5)  # Node positioning
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)

# Draw edge labels
edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")

# Save the graph
knowledge_graph_path = output_plot_dir / "knowledge_graph.png"
plt.savefig(knowledge_graph_path, bbox_inches="tight", dpi=300)
plt.show()

print(f"✅ Knowledge graph saved at: {knowledge_graph_path}")
