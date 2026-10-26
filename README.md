# RepoMind — AI-Powered GitHub Repository Intelligence

**Explore Python codebases with structure-aware retrieval and grounded AI explanations.**

RepoMind is a retrieval-augmented generation (RAG) application that accepts a public GitHub repository URL, extracts supported source files, builds a searchable code index, and answers natural-language questions using retrieved code as evidence.

Instead of treating source code as arbitrary text, RepoMind uses Python's Abstract Syntax Tree (AST) to identify classes, functions, methods, decorators, and selected module-level statements. It combines **FAISS semantic search** with **BM25 keyword search**, merges rankings through **Reciprocal Rank Fusion (RRF)**, and sends the retrieved evidence to **Claude** to generate source-grounded explanations.

## Screenshots
<img width="975" height="470" alt="image" src="https://github.com/user-attachments/assets/717a2dca-319b-4ac6-bdf3-6282bd33f2bc" />
<img width="975" height="490" alt="image" src="https://github.com/user-attachments/assets/8b9dcf69-010c-400a-92a3-119ec0a55fe2" />
<img width="975" height="463" alt="image" src="https://github.com/user-attachments/assets/e34f3370-5a11-4901-bb66-40be41df299b" />
<img width="975" height="418" alt="image" src="https://github.com/user-attachments/assets/10458059-dbe1-4607-bf9d-17e0e65feccb" />

## Features

- **Public GitHub repository ingestion:** Shallow-clones a repository and discovers supported files.
- **Python AST chunking:** Extracts top-level functions, classes, direct methods, decorators, imports, and selected assignments, with file paths and line numbers.
- **Jupyter notebook support:** Extracts code and Markdown cells from `.ipynb` files. Notebook cells use cell-based chunking in the current implementation.
- **Dense retrieval:** Generates Sentence Transformers embeddings and searches normalized vectors with FAISS `IndexFlatIP`.
- **Sparse retrieval:** Uses BM25 for keyword matches and code identifiers.
- **Hybrid retrieval:** Combines dense and sparse results using RRF.
- **Grounded code explanations:** Uses Anthropic Claude to answer with references to retrieved snippets.
- **Streamlit interface:** Repository URL onboarding, indexing status, chat, and expandable retrieved code sources.
- **Local command-line workflows:** Search and RAG pipeline scripts for testing without the UI.

## How it works

```text
Public GitHub repository URL
           |
           v
Shallow clone + file discovery
           |
           +--> Python (.py) --> AST-based chunks
           |
           +--> Notebooks (.ipynb) --> Code/Markdown cell chunks
           |
           v
Chunks + source metadata
           |
           +--> Sentence Transformers --> FAISS dense index
           |
           +--> Tokenization ---------> BM25 sparse index
           |                             |
           +----------- RRF ------------+
                        |
                        v
               Top-k source chunks
                        |
                        v
               Claude + grounded prompt
                        |
                        v
          Answer + source snippets and locations
```

The FAISS index stores vectors; the corresponding code snippets and metadata are kept separately and matched by index position. In the command-line workflow, the index and chunks can be saved to disk for reuse. The interactive repository-onboarding path currently builds indexes in the running session.

## Tech stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.12 (development environment) |
| Git ingestion | GitPython |
| Python parsing | Built-in `ast` |
| Notebook ingestion | `nbformat` |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatIP`) |
| Lexical search | `rank-bm25` |
| Rank fusion | Reciprocal Rank Fusion (RRF) |
| LLM | Anthropic Claude API |
| Interface | Streamlit |

## Project structure

```text
app/
├── chunking/
│   └── chunker.py             # AST chunks and oversized-chunk splitting
├── ingestion/
│   ├── github_loader.py       # Clone and load repository files
│   └── notebook_loader.py     # Notebook cell extraction (if separated)
├── rag/
│   └── answer_generator.py    # Grounded LLM generation
├── retrieval/
│   ├── embeddings.py          # Embedding utilities
│   ├── vector_store.py        # FAISS creation, retrieval and persistence
│   ├── bm25_retriever.py      # Sparse keyword retrieval
│   └── hybrid_retriever.py    # RRF-based fusion
├── proj_pipeline.py           # Ingestion-to-chunk integration
├── rag_pipeline.py            # CLI RAG Q&A
├── search_demo.py             # CLI hybrid retrieval demo
└── streamlit_app.py           # Streamlit web interface
.streamlit/
└── config.toml                # UI theme
requirements.txt
.env.example                    # Example values only, no secrets
README.md
```

Depending on your local cleanup, additional example or test files may be present. The layout above lists the core components rather than claiming every optional file is included.

## Getting started

### Prerequisites

- Python 3.12 (the development version used for this project)
- Git installed and available on your `PATH`
- An Anthropic API key for AI-generated answers
- Internet access for GitHub cloning, the initial embedding-model download, and Claude requests

### 1. Clone this project

```bash
git clone https://github.com/YOUR_USERNAME/RepoMind-AI.git
cd RepoMind-AI
```

Replace `YOUR_USERNAME` with the actual GitHub account after publishing.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and set your values:

```dotenv
ANTHROPIC_API_KEY=your_actual_key
ANTHROPIC_MODEL=your_valid_anthropic_model_id
```

The model ID must be a model available to your Anthropic API account. **Never commit `.env` or an API key.** Anthropic API usage can incur charges.

### 5. Launch RepoMind

Run this command **from the project root**:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL shown in your terminal (typically `http://localhost:8501`).

## Using the application

1. Paste a public GitHub repository URL, such as `https://github.com/pallets/click`.
2. Select **Analyze repository**. RepoMind discovers supported files, chunks code, embeds it, and builds hybrid retrieval indexes.
3. Once indexing succeeds, ask a question in the **AI Assistant** tab.
4. Expand **Sources** to inspect retrieved snippets, paths, and line ranges.
5. Use **Source Explorer** to search for classes, methods, or code concepts directly.

### Example questions

- Where is the `Context` class defined?
- What does `Command.make_context()` do?
- How does this codebase parse command-line arguments?
- Where is a particular exception class defined?
- Which functions relate to input validation?

For notebook-heavy machine learning repositories, you can also ask where model initialization, dataset loading, or training logic appears, provided those cells have been indexed.

## Command-line workflows

Run the retrieval demo:

```bash
python -m app.search_demo
```

Run the CLI RAG question-answering assistant:

```bash
python -m app.rag_pipeline
```

These scripts use the saved index in `data/index` created by the retrieval demo. If you change the embedding model, indexed repository, chunking logic, or metadata schema, rebuild the index to avoid mismatched results.


## Design choices

**Why AST chunking?** Arbitrary character splitting can separate a function or method from its surrounding logic. AST parsing exposes structural boundaries, making it possible to preserve meaningful code units and source metadata.

**Why FAISS?** FAISS offers lightweight local vector indexing with direct control over similarity search and normalization. It is a vector-search library rather than a full metadata database.

**Why BM25 as well?** Dense search helps with semantic questions, while BM25 can surface exact identifiers and rare symbols. RRF combines ranked results without directly adding incompatible raw scores.

**Why retain source metadata?** File paths, symbols, and line ranges help readers inspect the code used to support an answer. Citations are currently based on retrieved metadata and still require automated validation for production-grade trustworthiness.

## Current limitations

- Only **public GitHub repositories** are supported in the current UI.
- `.py` AST support is intentionally partial: nested definitions, all expression statements, and some class structures may be missed.
- Notebook code and Markdown are indexed by cell; notebook outputs, charts, and embedded images are **not** interpreted.
- Oversized source chunks use a line-based fallback rather than a tokenizer-aware or statement-aware algorithm.
- Repository-wide architecture or change-impact questions may require more evidence than the current top-k retrieval returns.
- Chat messages are visible in session history, but contextual follow-up resolution is not yet fully implemented.
- New repository indexing runs in the active Streamlit session and is not yet persisted automatically per repository.
- Public-facing deployment requires stronger Git cloning isolation, input and resource limits, secret scanning, and security review.
- Formal benchmarks and automatic citation verification are planned, not yet completed.



