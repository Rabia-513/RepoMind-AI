
from app.retrieval.vector_store import retrieve_code
from app.retrieval.bm25_retriever import BM25Retriever


class HybridRetriever:

    def __init__(self, model, index, chunks):
        self.model = model
        self.index = index
        self.chunks = chunks

        self.bm25 = BM25Retriever(chunks)

    def search(self, query, k=5):
        if k <= 0:
            raise ValueError("k must be positive")

        candidate_k = min(
            max(20, k * 4),
            len(self.chunks)
        )

        dense_results = retrieve_code(
            query,
            self.model,
            self.index,
            self.chunks,
            k=candidate_k
        )

        sparse_results = self.bm25.search(
            query,
            k=candidate_k
        )

        rrf_scores = {}

        # Combine both rankings
        for ranking in [dense_results, sparse_results]:

            for rank, result in enumerate(ranking, start=1):

                idx = result["chunk_id"]

                rrf_scores[idx] = (
                    rrf_scores.get(idx, 0.0)
                    + 1.0 / (60 + rank)
                )

        ranked_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True
        )

        return [
            {
                "chunk_id": idx,
                **self.chunks[idx],
                "rrf_score": rrf_scores[idx]
            }
            for idx in ranked_ids[:k]
        ]
