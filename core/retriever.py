# core/retriever.py
# FAISS-based vector retriever using Sentence-BERT

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDER_MODEL)


def build_index(corpus_docs: list, index_path: str, corpus_path: str, nlist: int = 100):
    """Encode corpus and build a FAISS IVF index."""
    print(f"Encoding {len(corpus_docs)} documents...")
    BATCH_SIZE = 512
    all_embeddings = []

    for i in range(0, len(corpus_docs), BATCH_SIZE):
        batch = corpus_docs[i:i + BATCH_SIZE]
        embs = embedder.encode(
            batch,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_embeddings.append(embs)

    corpus_embeddings = np.vstack(all_embeddings)
    dimension = corpus_embeddings.shape[1]

    # Use flat index if corpus is small, IVF otherwise
    if len(corpus_docs) < 1000:
        index = faiss.IndexFlatIP(dimension)
        index.add(corpus_embeddings)
    else:
        nlist = min(nlist, len(corpus_docs) // 10)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        index.nprobe = 10

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus_docs, f)

    print(f"Index built: {index.ntotal} vectors saved to {index_path}")
    return index, corpus_docs


def load_index(index_path: str, corpus_path: str):
    """Load existing FAISS index and corpus."""
    index = faiss.read_index(index_path)
    with open(corpus_path, "rb") as f:
        corpus_docs = pickle.load(f)
    return index, corpus_docs


def retrieve_top_k(query: str, index, corpus_docs: list, k: int = 5) -> list:
    """Retrieve top-k most relevant documents for a query."""
    qe = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    scores, indices = index.search(qe, k)
    return [
        {"document": corpus_docs[idx], "score": float(score)}
        for score, idx in zip(scores[0], indices[0]) if idx != -1
    ]