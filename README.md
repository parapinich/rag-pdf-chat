# 📄 Chat with Your Own PDF — RAG System

A **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF documents and ask questions about their content. Built with Python, FastAPI, LangChain, FAISS, and open-source HuggingFace models.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎥 Demo

📺 **[Watch Demo Video](https://drive.google.com/drive/folders/1nMn1185RunjcFTl106SpbVYthzS6X3ED?usp=sharing)**

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────────────────────────┐
│   Streamlit UI  │◄────────────►│          FastAPI Backend             │
│  (streamlit_app)│              │                                     │
└─────────────────┘              │  ┌──────────┐   ┌───────────────┐  │
                                 │  │ Guardrail│──►│  RAG Engine    │  │
                                 │  └──────────┘   │               │  │
                                 │                 │ ┌───────────┐ │  │
                                 │                 │ │PDF Loader │ │  │
                                 │                 │ ├───────────┤ │  │
                                 │                 │ │ Chunking  │ │  │
                                 │                 │ ├───────────┤ │  │
                                 │                 │ │ FAISS     │ │  │
                                 │                 │ ├───────────┤ │  │
                                 │                 │ │ LLM (T5)  │ │  │
                                 │                 │ └───────────┘ │  │
                                 │                 └───────────────┘  │
                                 │  ┌──────────────┐                  │
                                 │  │  Evaluation   │                  │
                                 │  │ (Hit Rate/MRR)│                  │
                                 │  └──────────────┘                  │
                                 └─────────────────────────────────────┘
```

See [architecture.md](architecture.md) for the full Mermaid diagram.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📤 PDF Upload | Upload any PDF and automatically index its content |
| 💬 Chat Interface | Ask natural-language questions about your document |
| ⚙️ Chunking Strategies | Choose from 3 strategies: `fixed`, `medium`, `sentence` |
| 🛡️ Prompt Guardrail | Blocks prompt injection, SQL injection, and invalid inputs |
| 🧪 Retrieval Evaluation | Measure retrieval quality with Hit Rate and MRR metrics |
| 📚 Source Transparency | View the exact source chunks used to generate answers |

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | FastAPI | Fast, async, auto-generated API docs |
| RAG Framework | LangChain | Mature ecosystem, great documentation |
| Vector Store | FAISS (CPU) | Lightweight, no external server needed |
| Embedding | `all-MiniLM-L6-v2` | Small (~80MB), fast, accurate |
| LLM | `google/flan-t5-base` | Open-source, free, ~1GB |
| Frontend | Streamlit | Pythonic, rapid prototyping |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/parapinich/rag-pdf-chat.git
cd rag-pdf-chat

# 2. Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
cp .env.example .env
```

### Running the Application

You need **two terminals** — one for the API, one for the UI:

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run ui/streamlit_app.py
```

Then open your browser:
- **Streamlit UI**: http://localhost:8501
- **API Docs (Swagger)**: http://localhost:8000/docs

---

## 📖 Usage

1. **Upload a PDF** — Use the sidebar to select a PDF file and choose a chunking strategy.
2. **Ask Questions** — Type a question in the chat input to get answers from your document.
3. **View Sources** — Expand the "Source Chunks" section to see which parts of the document were used.
4. **Run Evaluation** — Click "Run Evaluation" in the sidebar to measure retrieval quality.

---

## ⚙️ Chunking Strategies

| Strategy | Chunk Size | Best For |
|----------|-----------|----------|
| `fixed` | 500 chars | Precise retrieval, short documents |
| `medium` | 1000 chars | Balanced context, general use |
| `sentence` | Sentence-based | Natural text boundaries, academic papers |

---

## 🧪 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Hit Rate** | % of queries where at least one relevant chunk is in the top-k results |
| **MRR** | Mean Reciprocal Rank — measures how high the first relevant chunk ranks |

---

## 📁 Project Structure

```
rag-pdf-chat/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── architecture.md         # System architecture diagram
├── app/
│   ├── __init__.py
│   ├── config.py           # Application settings
│   ├── rag_engine.py       # Core RAG: chunking, embedding, retrieval, QA
│   ├── guardrail.py        # Input validation and safety checks
│   ├── evaluation.py       # Retrieval quality evaluation
│   └── api.py              # FastAPI REST endpoints
├── ui/
│   └── streamlit_app.py    # Streamlit chat interface
└── uploads/                # Uploaded PDF storage
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + document status |
| `POST` | `/upload` | Upload PDF & build vector index |
| `POST` | `/query` | Ask a question about the document |
| `POST` | `/evaluate` | Run retrieval evaluation |

Full interactive docs available at `http://localhost:8000/docs` (Swagger UI).

---

## 🛡️ Prompt Guardrail

The guardrail module protects against:
- **Empty queries** — Rejects blank or whitespace-only inputs
- **Oversized queries** — Limits input to 500 characters
- **Prompt injection** — Blocks "ignore previous instructions" and similar attacks
- **SQL injection** — Detects SQL keywords and patterns
- **Command injection** — Blocks shell command patterns

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) — RAG framework
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search by Meta
- [HuggingFace](https://huggingface.co/) — Open-source models
- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Streamlit](https://streamlit.io/) — Data app framework
