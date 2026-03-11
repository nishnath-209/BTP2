"""
rag/retriever.py

RAG retrieval using FAISS + SentenceTransformer.
Retrieves top-k clinically relevant chunks from therapy session knowledge base.
"""

import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("rag/index.faiss")

with open("rag/docs.pkl", "rb") as f:
    docs = pickle.load(f)


def retrieve(query: str, k: int = 3, debug: bool = True) -> list:
    """
    Retrieve top-k relevant clinical knowledge chunks for a given query.
    Returns a list of text strings.
    """

    if debug:
        print(f"\n[RAG] Query: {query}")

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = [docs[idx] for idx in indices[0]]

    if debug:
        print(f"[RAG] Retrieved {len(results)} chunks (distances: {distances[0].tolist()})")

    return results