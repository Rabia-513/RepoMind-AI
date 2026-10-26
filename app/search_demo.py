
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.proj_pipeline import process_repository

from app.retrieval.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)

from app.retrieval.hybrid_retriever import HybridRetriever


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FOLDER = "data/index"


def main():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    index_path = Path(INDEX_FOLDER) / "code.index"
    chunks_path = Path(INDEX_FOLDER) / "chunks.json"

    # Reuse previously saved indexes
    if index_path.exists() and chunks_path.exists():

        print("Loading saved FAISS index...")

        index, chunks = load_vector_store(
            INDEX_FOLDER
        )

    else:

        repo_url = "https://github.com/pallets/click.git"

        print("Processing GitHub repository...")

        chunks = process_repository(repo_url)

        if not chunks:
            raise ValueError("No chunks extracted")

        print("Total chunks:", len(chunks))

        # Code chunk embeddings
        texts = [
            chunk["content"]
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # Build FAISS index
        index = create_vector_store(embeddings)

        # Save index and metadata
        save_vector_store(index, chunks, INDEX_FOLDER)

    # Build hybrid search engine
    retriever = HybridRetriever(
        model,
        index,
        chunks
    )

    print("\nCode search engine ready!")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nAsk about the code: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = retriever.search(query, k=5)

        if not results:
            print("No matching code found.")
            continue

        print("\n===== RETRIEVED CODE =====")

        for result in results:

            print("\nFile:", result["file_path"])

            print(
                "Function:",
                result.get("qualified_name")
            )

            print(
                "Lines:",
                result["start_line"],
                "-",
                result["end_line"]
            )

            print(
                "RRF score:",
                round(result["rrf_score"], 5)
            )

            print("Code:")
            print(result["content"][:500])

            print("-" * 50)


if __name__ == "__main__":
    main()
