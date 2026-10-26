
from git import Repo
from pathlib import Path
import tempfile
import shutil
import time
import re
import nbformat
from urllib.parse import urlparse


MAX_FILE_SIZE = 1_000_000

IGNORED_FOLDERS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
}


# ---------------------------------------
# 1. VALIDATE AND CLONE REPOSITORY
# ---------------------------------------

def validate_repo_url(repo_url):
    parsed = urlparse(repo_url.strip())

    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ValueError("Only HTTPS github.com URLs are supported")

    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Invalid GitHub URL")

    if parsed.query or parsed.fragment:
        raise ValueError("Remove query parameters and fragments")

    parts = parsed.path.strip("/").split("/")

    if len(parts) != 2:
        raise ValueError("Use https://github.com/owner/repository")

    owner, repo = parts
    repo = repo.removesuffix(".git")

    pattern = r"[A-Za-z0-9_.-]+"

    if (
        not re.fullmatch(pattern, owner)
        or not re.fullmatch(pattern, repo)
        or owner in (".", "..")
        or repo in (".", "..")
    ):
        raise ValueError("Invalid repository name")

    return f"https://github.com/{owner}/{repo}.git"


def clone_repository(repo_url, destination):
    validated_url = validate_repo_url(repo_url)

    return Repo.clone_from(
        validated_url,
        destination,
        depth=1
    )


# ---------------------------------------
# 2. DISCOVER PYTHON AND NOTEBOOK FILES
# ---------------------------------------

def get_source_files(repo_path):
    repository_path = Path(repo_path)
    source_files = []

    for file_path in repository_path.rglob("*"):

        if any(
            part in IGNORED_FOLDERS
            for part in file_path.relative_to(repository_path).parts
        ):
            continue

        if file_path.suffix.lower() not in (".py", ".ipynb"):
            continue

        if file_path.is_symlink() or not file_path.is_file():
            continue

        try:
            if file_path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue

        source_files.append(file_path)

    return sorted(source_files)


# Keep compatibility with old code that uses this function.
def get_python_files(repo_path):
    return [
        p for p in get_source_files(repo_path)
        if p.suffix.lower() == ".py"
    ]


# ---------------------------------------
# 3. READ NORMAL PYTHON FILE
# ---------------------------------------

def read_python_file(file_path, repo_path):
    file_path = Path(file_path)
    repo_path = Path(repo_path)

    relative_path = file_path.relative_to(repo_path).as_posix()

    content = file_path.read_text(encoding="utf-8")

    return {
        "file_path": relative_path,
        "language": "python",
        "source_type": "python",
        "content": content,
    }


# ---------------------------------------
# 4. READ JUPYTER NOTEBOOK CELLS
# ---------------------------------------

def read_notebook(file_path, repo_path):
    file_path = Path(file_path)
    repo_path = Path(repo_path)

    relative_path = file_path.relative_to(repo_path).as_posix()

    notebook = nbformat.read(
        str(file_path),
        as_version=4
    )

    documents = []

    language_name = (
        notebook.metadata.get("kernelspec", {}).get("language")
        or notebook.metadata.get("language_info", {}).get("name")
        or "python"
    ).lower()

    for cell_number, cell in enumerate(
        notebook.cells,
        start=1
    ):
        if cell.cell_type not in ("code", "markdown"):
            continue

        # Preserve original source, including line spacing.
        content = cell.source

        if not content.strip():
            continue

        documents.append({
            "file_path": relative_path,
            "language": (
                language_name
                if cell.cell_type == "code"
                else "markdown"
            ),
            "source_type": "notebook",
            "cell_number": cell_number,
            "cell_type": cell.cell_type,
            "content": content,
        })

    return documents


# ---------------------------------------
# 5. LOAD COMPLETE REPOSITORY
# ---------------------------------------

def load_repository(repo_url):

    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True
    ) as temp_dir:

        repo = None
        documents = []

        try:
            repo = clone_repository(repo_url, temp_dir)

            commit_sha = repo.head.commit.hexsha
            print("Indexed commit:", commit_sha)

            source_files = get_source_files(temp_dir)

            for file_path in source_files:

                try:
                    if file_path.suffix.lower() == ".py":
                        extracted = [
                            read_python_file(file_path, temp_dir)
                        ]

                    elif file_path.suffix.lower() == ".ipynb":
                        extracted = read_notebook(
                            file_path,
                            temp_dir
                        )

                    else:
                        continue

                    for document in extracted:
                        document["commit_sha"] = commit_sha
                        documents.append(document)

                except (
                    UnicodeDecodeError,
                    OSError,
                    ValueError,
                    nbformat.reader.NotJSONError
                ) as error:
                    print(f"Skipping {file_path}: {error}")

        finally:
            if repo is not None:
                repo.close()

            git_folder = Path(temp_dir) / ".git"

            if git_folder.exists():
                for attempt in range(3):
                    try:
                        shutil.rmtree(git_folder)
                        break

                    except PermissionError:
                        if attempt == 2:
                            print(
                                "Warning: Some temporary Git "
                                "metadata could not be removed."
                            )
                        else:
                            time.sleep(0.5)

        return documents


# ---------------------------------------
# 6. TEST
# ---------------------------------------

if __name__ == "__main__":

    repo_url = "https://github.com/pallets/click.git"

    documents = load_repository(repo_url)

    python_docs = [
        d for d in documents
        if d["source_type"] == "python"
    ]

    notebook_docs = [
        d for d in documents
        if d["source_type"] == "notebook"
    ]

    print("Total documents:", len(documents))
    print("Python files:", len(python_docs))
    print("Notebook cells:", len(notebook_docs))

    for document in documents[:5]:
        print("\nFile:", document["file_path"])
        print("Type:", document["source_type"])

        if document["source_type"] == "notebook":
            print("Cell:", document["cell_number"])
            print("Cell type:", document["cell_type"])

        print("Preview:", document["content"][:150])
