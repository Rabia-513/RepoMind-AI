
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.retrieval.vector_store import load_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.answer_generator import generate_answer


def main():

    index_folder = Path("data/index")

    if not (index_folder / "code.index").exists():
        raise FileNotFoundError(
            "FAISS index missing. Run app.search_demo first."
        )

    print("Loading embedding model...")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading vector index...")
    index, chunks = load_vector_store(index_folder)

    print("Initializing hybrid retriever...")
    retriever = HybridRetriever(model, index, chunks)

    print("\nGitHub RAG Assistant Ready!")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nAsk about the code: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = retriever.search(question, k=5)

        print("\nRetrieved sources:")

        for i, result in enumerate(results, start=1):
            print(
                f"[Source {i}] "
                f"{result['file_path']}:"
                f"{result['start_line']}-"
                f"{result['end_line']}"
            )

        answer = generate_answer(question, results)

        print("\nAI Answer:\n")
        print(answer)


if __name__ == "__main__":
    main()

