
import re
import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text):
    # Split camelCase and separate punctuation
    text = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        text
    )

    return re.findall(
        r"[a-zA-Z0-9_]+",
        text.lower()
    )


class BM25Retriever:

    def __init__(self, chunks):
        if not chunks:
            raise ValueError("No chunks provided")

        self.chunks = chunks

        self.documents = [
            (
                f"{chunk['file_path']} "
                f"{chunk.get('qualified_name') or ''} "
                f"{chunk['content']}"
            )
            for chunk in chunks
        ]

        self.tokenized_documents = [
            tokenize(doc)
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_documents
        )

    def search(self, query, k=5):
        if not query.strip():
            return []

        if k <= 0:
            raise ValueError("k must be positive")

        scores = self.bm25.get_scores(
            tokenize(query)
        )

        ranked_indices = np.argsort(scores)[::-1]

        results = []

        for idx in ranked_indices:
            if len(results) >= k:
                break

            score = float(scores[idx])

            if score <= 0:
                continue

            results.append({
                "chunk_id": int(idx),
                **self.chunks[int(idx)],
                "bm25_score": score
            })

        return results
