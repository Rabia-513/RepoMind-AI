
import faiss
import numpy as np
import json
from pathlib import Path


def create_vector_store(embeddings):
    vectors = np.asarray(embeddings, dtype=np.float32).copy()

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a nonempty 2D array")

    if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding vectors")

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def retrieve_code(query, model, index, chunks, k=5):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if k <= 0:
        raise ValueError("k must be positive")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk counts do not match")

    query_vector = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_vector = np.asarray(
        query_vector, dtype=np.float32
    )

    faiss.normalize_L2(query_vector)

    if query_vector.shape[1] != index.d:
        raise ValueError("Embedding dimensions do not match")

    scores, indices = index.search(
        query_vector,
        min(k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "chunk_id": int(idx),
            **chunks[int(idx)],
            "similarity_score": float(score)
        })

    return results


def save_vector_store(index, chunks, folder="data/index"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk counts do not match")

    faiss.write_index(index, str(folder / "code.index"))

    with open(folder / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def load_vector_store(folder="data/index"):
    folder = Path(folder)

    index = faiss.read_index(str(folder / "code.index"))

    with open(folder / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if index.ntotal != len(chunks):
        raise ValueError("Saved index and chunks are inconsistent")

    return index, chunks
