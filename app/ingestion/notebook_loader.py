
from pathlib import Path
import nbformat


def read_notebook(file_path, repo_path):
    file_path = Path(file_path)
    repo_path = Path(repo_path)

    notebook = nbformat.read(
        str(file_path),
        as_version=4
    )

    relative_path = file_path.relative_to(
        repo_path
    ).as_posix()

    documents = []

    for cell_number, cell in enumerate(
        notebook.cells, start=1
    ):
        content = cell.source.strip()

        if not content:
            continue

        if cell.cell_type not in ("code", "markdown"):
            continue

        documents.append({
            "file_path": relative_path,
            "language": "python" if cell.cell_type == "code" else "markdown",
            "content": content,
            "cell_number": cell_number,
            "cell_type": cell.cell_type,
            "source_type": "notebook"
        })

    return documents
