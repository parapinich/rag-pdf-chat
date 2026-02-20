"""
Streamlit UI — Simple chat interface for the RAG PDF system.

Features:
  - Sidebar: Upload PDF, select chunking strategy, run evaluation
  - Main area: Chat-style Q&A interface
  - Expandable source chunks for transparency
"""

import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: 600;
    }
    .status-error {
        color: #dc3545;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False
if "doc_info" not in st.session_state:
    st.session_state.doc_info = {}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def upload_pdf(file, strategy: str) -> dict:
    """Upload a PDF to the backend API."""
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    data = {"strategy": strategy}
    resp = requests.post(f"{API_URL}/upload", files=files, data=data, timeout=120)
    resp.raise_for_status()
    return resp.json()


def query_api(question: str) -> dict:
    """Send a question to the backend API."""
    resp = requests.post(
        f"{API_URL}/query",
        json={"question": question},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def run_evaluation() -> dict:
    """Run retrieval evaluation via the backend API."""
    resp = requests.post(f"{API_URL}/evaluate", timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/pdf-2.png", width=64)
    st.title("📄 PDF Chat")
    st.markdown("---")

    # API Status
    api_online = check_api_health()
    if api_online:
        st.markdown("🟢 **API Status:** <span class='status-success'>Online</span>", unsafe_allow_html=True)
    else:
        st.markdown("🔴 **API Status:** <span class='status-error'>Offline</span>", unsafe_allow_html=True)
        st.warning("Start the API server first:\n```\nuvicorn app.api:app --reload\n```")

    st.markdown("---")

    # Upload Section
    st.subheader("📤 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF document to chat with.",
    )

    strategy = st.selectbox(
        "Chunking Strategy",
        options=["fixed", "medium", "sentence"],
        index=0,
        help=(
            "**fixed**: Small chunks (500 chars) — more precise retrieval\n\n"
            "**medium**: Larger chunks (1000 chars) — more context per chunk\n\n"
            "**sentence**: Sentence-based splitting — natural boundaries"
        ),
    )

    if st.button("🚀 Upload & Process", disabled=not api_online, use_container_width=True):
        if uploaded_file is not None:
            with st.spinner("Processing PDF... This may take a moment."):
                try:
                    result = upload_pdf(uploaded_file, strategy)
                    st.session_state.document_loaded = True
                    st.session_state.doc_info = result
                    st.session_state.chat_history = []
                    st.success(
                        f"✅ **{result['filename']}** loaded!\n\n"
                        f"Chunks: **{result['num_chunks']}** | "
                        f"Strategy: **{result['strategy']}**"
                    )
                except requests.HTTPError as e:
                    st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")
        else:
            st.warning("Please select a PDF file first.")

    # Document Info
    if st.session_state.document_loaded:
        st.markdown("---")
        st.subheader("📊 Document Info")
        info = st.session_state.doc_info
        st.markdown(f"📁 **File:** {info.get('filename', 'N/A')}")
        st.markdown(f"📦 **Chunks:** {info.get('num_chunks', 'N/A')}")
        st.markdown(f"⚙️ **Strategy:** {info.get('strategy', 'N/A')}")

    # Evaluation Section
    if st.session_state.document_loaded:
        st.markdown("---")
        st.subheader("🧪 Retrieval Evaluation")
        if st.button("Run Evaluation", use_container_width=True):
            with st.spinner("Evaluating retrieval quality..."):
                try:
                    eval_result = run_evaluation()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hit Rate", f"{eval_result['hit_rate']:.1%}")
                    with col2:
                        st.metric("MRR", f"{eval_result['mrr']:.3f}")

                    with st.expander("📋 Evaluation Details"):
                        for detail in eval_result["details"]:
                            icon = "✅" if detail["hit"] else "❌"
                            st.markdown(
                                f"{icon} **Q:** {detail['question']}\n"
                                f"  - Rank: {detail.get('first_relevant_rank', 'N/A')}"
                            )
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")

    # Clear Chat
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------

st.markdown("<p class='main-header'>💬 Chat with Your PDF</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>"
    "Upload a PDF document and ask questions about its content. "
    "Powered by RAG (Retrieval-Augmented Generation)."
    "</p>",
    unsafe_allow_html=True,
)

if not st.session_state.document_loaded:
    st.info("👈 Upload a PDF document from the sidebar to get started!")

# Display chat history
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant"):
        st.write(entry["answer"])

        if entry.get("sources"):
            with st.expander(f"📚 Source Chunks ({len(entry['sources'])})"):
                for i, src in enumerate(entry["sources"], 1):
                    st.info(f"**Chunk {i}** (Page {src.get('page', 'N/A')})\n\n{src['content'][:300]}{'...' if len(src['content']) > 300 else ''}")

# Chat input
if question := st.chat_input("Ask a question about your document...", disabled=not st.session_state.document_loaded):
    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_api(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.write(answer)

                if sources:
                    with st.expander(f"📚 Source Chunks ({len(sources)})"):
                        for i, src in enumerate(sources, 1):
                            st.info(f"**Chunk {i}** (Page {src.get('page', 'N/A')})\n\n{src['content'][:300]}{'...' if len(src['content']) > 300 else ''}")

                # Save to history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                })

            except requests.HTTPError as e:
                error_detail = e.response.json().get("detail", str(e))
                st.error(f"⚠️ {error_detail}")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
