
from app.ingestion.github_loader import load_repository
from app.chunking.chunker import (
    chunk_python_code,
    split_large_chunks
)


def make_cell_chunks(document):
    """Create fallback chunks for notebook cells."""

    content = document["content"]
    lines = content.splitlines()

    if not lines:
        return []

    chunks = []

    # Split notebook cell text into manageable pieces.
    for start in range(0, len(lines), 50):
        piece = lines[start:start + 50]

        chunks.append({
            "file_path": document["file_path"],
            "qualified_name": (
                f"cell_{document['cell_number']}"
            ),
            "chunk_type": document["cell_type"],
            "cell_number": document["cell_number"],
            "cell_type": document["cell_type"],
            "source_type": "notebook",
            "commit_sha": document["commit_sha"],
            "start_line": start + 1,
            "end_line": start + len(piece),
            "content": "\n".join(piece),
        })

    return chunks


def process_repository(repo_url):

    documents = load_repository(repo_url)
    all_chunks = []

    for document in documents:

        if document["source_type"] == "notebook":
            # A safe cell-based baseline also handles
            # notebook magics and Markdown.
            chunks = make_cell_chunks(document)

        else:
            try:
                chunks = chunk_python_code(
                    document["content"],
                    document["file_path"]
                )

                chunks = split_large_chunks(
                    chunks,
                    max_lines=50
                )

            except SyntaxError as error:
                print(
                    f"Skipping {document['file_path']}: "
                    f"{error}"
                )
                continue

        for chunk in chunks:
            chunk["commit_sha"] = document["commit_sha"]
            chunk.setdefault(
                "source_type",
                document["source_type"]
            )

        all_chunks.extend(chunks)

    return all_chunks
