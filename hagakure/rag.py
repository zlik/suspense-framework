import os
import pickle
import warnings

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

warnings.simplefilter("ignore")  # Suppress unwanted warnings

# Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define FAISS index file path
FAISS_INDEX_FILE = "faiss_index.pkl"

# Initialize or Load FAISS Index
if os.path.exists(FAISS_INDEX_FILE):
    with open(FAISS_INDEX_FILE, "rb") as f:
        faiss_index, doc_store = pickle.load(f)
    print(f"[INFO] Loaded existing FAISS index with {len(doc_store)} documents.")
else:
    faiss_index = faiss.IndexFlatL2(384)  # 384 is the embedding size for MiniLM
    doc_store = []  # Store documents as list
    print("[INFO] Initialized new FAISS index.")


def save_faiss():
    """Save FAISS index and document store to disk."""
    with open(FAISS_INDEX_FILE, "wb") as f:
        pickle.dump((faiss_index, doc_store), f)
    print("[DEBUG] FAISS index saved.")


def add_document(text):
    """Embeds a document and adds it to the FAISS index."""
    global faiss_index, doc_store
    embedding = (
        embedding_model.encode(text, normalize_embeddings=True)
        .astype(np.float32)
        .reshape(1, -1)
    )

    faiss_index.add(embedding)
    doc_store.append(text)
    save_faiss()

    print(f"[DEBUG] Document added: '{text[:50]}...' (Embedding: {embedding.shape})")


def retrieve_context(query, top_k=3):
    """Retrieves the most relevant documents for a given query, with debug output."""
    debug_info = f"[INFO] Retrieving context for query: '{query}'\n"

    if len(doc_store) == 0:
        debug_info += "[WARNING] No documents in FAISS index.\n"
        print(debug_info)
        return "", debug_info

    query_embedding = (
        embedding_model.encode(query, normalize_embeddings=True)
        .astype(np.float32)
        .reshape(1, -1)
    )
    distances, indices = faiss_index.search(query_embedding, top_k)

    debug_info += f"[DEBUG] Query embedding shape: {query_embedding.shape}\n"
    debug_info += f"[DEBUG] Retrieved indices: {indices.tolist()}\n"
    debug_info += f"[DEBUG] Distances: {distances.tolist()}\n"

    retrieved_texts = [doc_store[i] for i in indices[0] if i < len(doc_store)]
    retrieved_context = "\n\n".join(retrieved_texts)

    debug_info += f"[INFO] Retrieved documents: {len(retrieved_texts)}\n"
    for i, text in enumerate(retrieved_texts):
        debug_info += f"  {i+1}. {text[:100]}...\n"

    print(debug_info)  # Print debug info for logging
    return retrieved_context, debug_info  # Return retrieved context and debug info
