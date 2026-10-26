
import ast


# --------------------------------------------------
# 1. EXTRACT STARTING LINE INCLUDING DECORATORS
# --------------------------------------------------

def get_start_line(node):
    """
    Return the first line of a function or class,
    including its decorators when present.
    """

    start_line = node.lineno

    if hasattr(node, "decorator_list"):
        if node.decorator_list:
            start_line = min(
                [node.lineno] +
                [d.lineno for d in node.decorator_list]
            )

    return start_line


# --------------------------------------------------
# 2. EXTRACT CODE CHUNKS USING AST
# --------------------------------------------------

def chunk_python_code(source_code, file_path):

    tree = ast.parse(source_code)
    lines = source_code.splitlines()

    all_chunks = []

    # Loop through top-level statements
    for node in tree.body:

        # ------------------------------------------
        # STANDALONE FUNCTIONS
        # ------------------------------------------
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            start_line = get_start_line(node)
            end_line = node.end_lineno

            content = "\n".join(
                lines[start_line - 1:end_line]
            )

            all_chunks.append({
                "file_path": file_path,
                "qualified_name": node.name,
                "chunk_type": "function",
                "start_line": start_line,
                "end_line": end_line,
                "content": content
            })

        # ------------------------------------------
        # CLASSES AND METHODS
        # ------------------------------------------
        elif isinstance(node, ast.ClassDef):

            class_start = get_start_line(node)

            # Capture the class declaration and docstring.
            # Stop before the first substantive class
            # statement after the optional docstring.
            if node.body:

                first = node.body[0]

                has_docstring = (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                )

                if has_docstring:
                    class_end = first.end_lineno
                else:
                    class_end = first.lineno - 1

                # Handle one-line class definitions.
                if class_end < node.lineno:
                    class_end = first.end_lineno

            else:
                class_end = node.end_lineno

            class_content = "\n".join(
                lines[class_start - 1:class_end]
            )

            all_chunks.append({
                "file_path": file_path,
                "qualified_name": node.name,
                "chunk_type": "class",
                "start_line": class_start,
                "end_line": class_end,
                "content": class_content
            })

            # Extract direct methods of this class
            for method in node.body:

                if isinstance(
                    method,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                ):

                    start_line = get_start_line(method)
                    end_line = method.end_lineno

                    content = "\n".join(
                        lines[start_line - 1:end_line]
                    )

                    all_chunks.append({
                        "file_path": file_path,
                        "qualified_name": (
                            f"{node.name}.{method.name}"
                        ),
                        "method_name": method.name,
                        "chunk_type": "method",
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": content
                    })

            # Preserve direct class attributes too
            for statement in node.body:

                if isinstance(
                    statement,
                    (ast.Assign, ast.AnnAssign)
                ):

                    start_line = statement.lineno
                    end_line = statement.end_lineno

                    content = "\n".join(
                        lines[start_line - 1:end_line]
                    )

                    all_chunks.append({
                        "file_path": file_path,
                        "qualified_name": node.name,
                        "chunk_type": "class_attribute",
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": content
                    })

        # ------------------------------------------
        # MODULE-LEVEL IMPORTS AND VARIABLES
        # ------------------------------------------
        elif isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.Assign,
                ast.AnnAssign
            )
        ):

            start_line = node.lineno
            end_line = node.end_lineno

            content = "\n".join(
                lines[start_line - 1:end_line]
            )

            all_chunks.append({
                "file_path": file_path,
                "qualified_name": None,
                "chunk_type": "top_level",
                "start_line": start_line,
                "end_line": end_line,
                "content": content
            })

    # Sort chunks according to source position
    all_chunks.sort(
        key=lambda chunk: chunk["start_line"]
    )

    return all_chunks


# --------------------------------------------------
# 3. SPLIT OVERSIZED CHUNKS
# --------------------------------------------------

def split_large_chunks(chunks, max_lines=50):

    if max_lines <= 0:
        raise ValueError(
            "max_lines must be greater than zero"
        )

    final_chunks = []

    for chunk in chunks:

        lines = chunk["content"].splitlines()

        # Keep small chunks unchanged
        if len(lines) <= max_lines:
            final_chunks.append(chunk)
            continue

        # Split oversized chunks
        for i in range(0, len(lines), max_lines):

            piece = lines[i:i + max_lines]

            new_start = chunk["start_line"] + i
            new_end = new_start + len(piece) - 1

            smaller_chunk = {
                **chunk,
                "content": "\n".join(piece),
                "start_line": new_start,
                "end_line": new_end,
                "part_number": (i // max_lines) + 1
            }

            final_chunks.append(smaller_chunk)

    return final_chunks


# --------------------------------------------------
# 4. TEST THE CHUNKER
# --------------------------------------------------

if __name__ == "__main__":

    code = '''
import os
from pathlib import Path

DATABASE_URL = "sqlite:///database.db"
MAX_CONNECTIONS = 10

class User:
    """Represents an application user."""

    role = "member"

    @property
    def name(self):
        return self._name

    @staticmethod
    def validate_email(email):
        return "@" in email


def connect_database():
    return DATABASE_URL
'''

    chunks = chunk_python_code(
        code,
        "example.py"
    )

    print("\n===== AST CHUNKING TEST =====")
    print("Total chunks:", len(chunks))

    for chunk in chunks:

        print("\nName:", chunk["qualified_name"])
        print("Type:", chunk["chunk_type"])

        print(
            "Lines:",
            chunk["start_line"],
            "-",
            chunk["end_line"]
        )

        print("Content:")
        print(chunk["content"])
        print("-" * 40)

    # ------------------------------------------
    # LARGE CHUNK TEST
    # ------------------------------------------

    print("\n===== LARGE CHUNK TEST =====")

    sample_chunks = [
        {
            "file_path": "large.py",
            "qualified_name": "calculate",
            "chunk_type": "function",
            "start_line": 10,
            "end_line": 119,
            "content": "\n".join(
                f"line {i}" for i in range(110)
            )
        }
    ]

    result = split_large_chunks(
        sample_chunks,
        max_lines=40
    )

    for chunk in result:

        print(
            chunk["qualified_name"],
            chunk["start_line"],
            chunk["end_line"],
            len(chunk["content"].splitlines())
        )
