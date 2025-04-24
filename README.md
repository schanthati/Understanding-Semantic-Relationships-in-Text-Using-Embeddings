# Understanding-Semantic-Relationships-in-Text-Using-Embeddings
# Understanding Semantic Relationships in Text Using Embeddings

# Using OpenAI's Embedding Model: text-embedding-ada-002
# This script compares semantic similarity between texts using cosine similarity

import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Step 0: Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

# Step 1: Define your input texts
texts = [
    "How do I bake a chocolate cake?",
    "What's the best recipe for a chocolate cake?",
    "What is quantum mechanics?"
]

# Step 2: Function to get text embeddings from OpenAI

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Step 3: Generate embeddings for all input texts
embeddings = [get_embedding(text) for text in texts]

# Step 4: Compute cosine similarity between all pairs
similarity_matrix = cosine_similarity(embeddings)

# Step 5: Display the similarity scores
print("\nSemantic Similarity Scores:\n")
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print(f"Similarity between:\n- '{texts[i]}'\n- '{texts[j]}'\n= {similarity_matrix[i][j]:.2f}\n")

# Example Output:
# Shows high similarity between cake-related queries and low similarity with unrelated scientific query

# Optional: To use HuggingFace SentenceTransformers instead (no API needed), replace the embedding section with:
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(texts)
