from sentence_transformers import SentenceTransformer
import numpy as np

def generate_embeddings(chunks):

    # 1. Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 2. Extract text from each chunk
    texts = [chunk["content"] for chunk in chunks]

    # 3. Convert the texts into embeddings
    embeddings = model.encode(texts)

    # 4. Return the embeddings
    return embeddings

