
import re
from urllib.parse import urlparse

import streamlit as st
from sentence_transformers import SentenceTransformer

from app.proj_pipeline import process_repository
from app.retrieval.vector_store import create_vector_store
from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.answer_generator import generate_answer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(
    page_title="RepoMind | AI Code Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown("""
<style>
/* Decoration only: widget colors come from .streamlit/config.toml */
.stApp { background: #0B1020; }
[data-testid="stHeader"] { background: #0B1020 !important; }
[data-testid="stSidebar"] { background: #101B2D !important; border-right: 1px solid #2C3952; }
.block-container { max-width: 1180px; padding-top: 2.25rem; }
.stApp, .stApp p, .stApp label, .stApp span, .stApp li { color: #E7EDFA; }
[data-testid="stSidebar"] * { color: #E7EDFA; }
[data-testid="stMetric"] { background: #172238; border: 1px solid #35445F; padding: 18px; border-radius: 16px; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] * { color: #B7C7E4 !important; }
.stButton > button { background: #1D2B45 !important; color: #F7F8FF !important; border: 1px solid #43557B !important; border-radius: 12px; }
.stButton > button:hover { background: #2B3E60 !important; border-color: #A78BFA !important; }
.stButton > button p { color: #F7F8FF !important; }
[data-testid="stFormSubmitButton"] button { background: #7357D5 !important; color: #FFFFFF !important; border: 1px solid #9B87F5 !important; }
[data-testid="stFormSubmitButton"] button p { color: #FFFFFF !important; }
[data-testid="stTabs"] button p { color: #DCE5FC !important; }
[data-testid="stTabs"] button[aria-selected="true"] p { color: #C4B5FD !important; }
[data-testid="stExpander"] { background: #142038; border: 1px solid #314363; border-radius: 12px; }
[data-testid="stChatMessage"] { background: #142038; border: 1px solid #314363; border-radius: 15px; }
[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { background: #1C2B43 !important; color: #F8FAFF !important; caret-color: #C4B5FD; }
[data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { color: #B2C0D7 !important; opacity: 1; }
[data-testid="stCaptionContainer"] * { color: #AAB9D3 !important; }
.hero { text-align: center; padding: 46px 0 28px; }
.kicker { color: #BCACFF; font-size: 12px; letter-spacing: 2px; font-weight: 750; }
.hero h1 { color: #F7F8FF !important; font-size: clamp(32px, 5vw, 52px); margin: 14px 0; }
.hero p { color: #C4CDE0 !important; font-size: 16px; max-width: 650px; margin: 0 auto; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "repo_url": None,
    "retriever": None,
    "chunks": [],
    "files_count": 0,
    "messages": [],
    "pending_question": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# VALIDATE GITHUB URL
# --------------------------------------------------

def validate_github_url(url):
    parsed = urlparse(url.strip())

    if parsed.scheme != "https":
        raise ValueError("Use an HTTPS GitHub URL.")

    if parsed.hostname != "github.com":
        raise ValueError("Only github.com repositories are supported.")

    if parsed.username or parsed.password or parsed.port:
        raise ValueError("Unsupported GitHub URL format.")

    parts = parsed.path.strip("/").split("/")

    if len(parts) != 2:
        raise ValueError(
            "Use https://github.com/owner/repository"
        )

    owner, repo = parts
    repo = repo.removesuffix(".git")

    valid = r"[A-Za-z0-9_.-]+"

    if (
        not re.fullmatch(valid, owner)
        or not re.fullmatch(valid, repo)
        or owner in (".", "..")
        or repo in (".", "..")
    ):
        raise ValueError("Invalid owner or repository name.")

    if parsed.query or parsed.fragment:
        raise ValueError("Remove URL query or fragment.")

    return f"https://github.com/{owner}/{repo}.git"


# --------------------------------------------------
# INDEX A NEW REPOSITORY
# --------------------------------------------------

def analyze_repository(url):
    repo_url = validate_github_url(url)

    with st.status("Analyzing repository...", expanded=True) as status:
        status.write("1/4 — Reading Python files and notebook cells; preparing chunks")
        # This shared pipeline handles .py and .ipynb (including Markdown cells).
        chunks = process_repository(repo_url)
        if not chunks:
            raise ValueError("No supported source code or notebook cells found.")
        file_count = len({chunk["file_path"] for chunk in chunks})
        status.write(f"Found {file_count} files and {len(chunks)} searchable chunks")

        status.write("2/4 — Generating source embeddings")
        model = get_embedding_model()
        texts = [chunk["content"] for chunk in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        status.write("3/4 — Creating FAISS index")
        index = create_vector_store(embeddings)
        status.write("4/4 — Creating BM25 and hybrid search")
        retriever = HybridRetriever(model, index, chunks)
        status.update(label="Repository successfully indexed", state="complete", expanded=False)

    st.session_state.repo_url = repo_url
    st.session_state.retriever = retriever
    st.session_state.chunks = chunks
    st.session_state.files_count = file_count
    st.session_state.messages = []
    st.session_state.pending_question = None

# --------------------------------------------------
# DISPLAY SOURCES
# --------------------------------------------------

def show_sources(sources):
    for i, source in enumerate(sources, 1):
        name = source.get("qualified_name") or "Module code"

        with st.expander(
            f"Source {i} · {name} · "
            f"{source['file_path']}"
        ):
            st.caption(
                f"Lines {source['start_line']}–"
                f"{source['end_line']}"
            )

            st.caption(
                f"Chunk type: {source.get('chunk_type', 'code')}"
                + (f" · Notebook cell {source['cell_number']}" if source.get('cell_number') else "")
            )

            st.code(source["content"], language=("markdown" if source.get("cell_type") == "markdown" else "python"))


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🧠 RepoMind")
    st.caption("AI CODE INTELLIGENCE")

    st.divider()

    if st.session_state.repo_url:
        st.success("Repository indexed")
        st.caption(st.session_state.repo_url)
    else:
        st.info("No repository loaded")

    st.divider()

    st.markdown("#### Retrieval stack")
    st.write("FAISS · Semantic search")
    st.write("BM25 · Keyword search")
    st.write("RRF · Hybrid ranking")
    st.write("Claude · Answer generation")

    st.divider()

    top_k = st.slider(
        "Sources per answer",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("Load another repository", use_container_width=True):
        st.session_state.repo_url = None
        st.session_state.retriever = None
        st.session_state.chunks = []
        st.session_state.files_count = 0
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# --------------------------------------------------
# SCREEN 1: REPOSITORY ONBOARDING
# --------------------------------------------------

if st.session_state.retriever is None:

    st.markdown("""
    <div class="hero">
        <div class="kicker">AI-POWERED CODE INTELLIGENCE</div>
        <h1>Understand Python repositories and notebooks.</h1>
        <p>
            Paste a public GitHub repository URL.
            RepoMind analyzes the code and turns it into
            a searchable AI knowledge base.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1, 2.5, 1])

    with center:
        with st.container(border=True):
            st.subheader("Connect your repository")

            with st.form("repo_form"):
                url = st.text_input(
                    "GitHub repository URL",
                    placeholder=(
                        "https://github.com/pallets/click"
                    )
                )

                submitted = st.form_submit_button(
                    "✨ Analyze repository",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                try:
                    analyze_repository(url)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

            st.caption(
                "Public GitHub repositories only · "
                "Python (.py) and Jupyter (.ipynb) supported"
            )

    st.divider()

    a, b, c = st.columns(3)

    with a:
        st.markdown("### 01 · Ingest")
        st.write("Read Python scripts and notebook cells.")

    with b:
        st.markdown("### 02 · Index")
        st.write("AST chunking, embeddings, FAISS and BM25.")

    with c:
        st.markdown("### 03 · Explore")
        st.write("Ask natural-language questions with citations.")

    st.stop()


# --------------------------------------------------
# SCREEN 2: REPOSITORY DASHBOARD
# --------------------------------------------------

st.markdown("""
<div class="kicker">YOUR REPOSITORY IS READY</div>
<h1>Explore your codebase</h1>
""", unsafe_allow_html=True)

st.caption(st.session_state.repo_url)

m1, m2, m3 = st.columns(3)

m1.metric(
    "Source files",
    st.session_state.files_count
)

m2.metric(
    "Indexed chunks",
    len(st.session_state.chunks)
)

m3.metric(
    "Search engine",
    "Hybrid RAG"
)

chat_tab, sources_tab = st.tabs([
    "💬 AI Assistant",
    "🔎 Source Explorer"
])


# --------------------------------------------------
# CHAT TAB
# --------------------------------------------------

with chat_tab:

    if not st.session_state.messages:

        st.subheader("Start exploring")

        suggestions = [
            "Explain the overall architecture of this repository.",
            "Where is the main entry point?",
            "What are the most important classes?",
            "How does this repository handle errors?"
        ]

        cols = st.columns(2)

        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(
                    suggestion,
                    key=f"prompt_{i}",
                    use_container_width=True
                ):
                    st.session_state.pending_question = suggestion

    # Show previous conversation
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                show_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask anything about this repository..."
    )

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching codebase..."):
                    results = st.session_state.retriever.search(
                        question,
                        k=top_k
                    )

                with st.spinner("Generating grounded answer..."):
                    answer = generate_answer(
                        question,
                        results
                    )

                st.markdown(answer)
                show_sources(results)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": results
                })

            except Exception as exc:
                st.error(f"Unable to answer: {exc}")


# --------------------------------------------------
# SOURCE EXPLORER
# --------------------------------------------------

with sources_tab:

    st.subheader("Search indexed code")

    search_query = st.text_input(
        "Find a function, class, or code concept",
        placeholder="e.g. authentication, Context, parse_args"
    )

    if search_query.strip():
        results = st.session_state.retriever.search(
            search_query,
            k=10
        )

        st.write(f"{len(results)} matching chunks")
        show_sources(results)

    else:
        st.info(
            "Enter a symbol or description to search "
            "the indexed repository."
        )
