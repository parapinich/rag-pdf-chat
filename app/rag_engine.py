"""
Core RAG Engine — handles PDF loading, chunking, embedding, and retrieval.

Supports multiple chunking strategies:
  - fixed: RecursiveCharacterTextSplitter with small chunks (500 chars)
  - medium: RecursiveCharacterTextSplitter with medium chunks (1000 chars)
  - sentence: Split by sentences using NLTK-based splitter

Uses FAISS for vector similarity search and HuggingFace models for
both embeddings and text generation.
"""

import os
from pathlib import Path
from typing import Optional

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

from app.config import settings


# ---------------------------------------------------------------------------
# Chunking Strategies
# ---------------------------------------------------------------------------

def get_text_splitter(strategy: str):
    """
    Return a text splitter based on the chosen strategy.

    Args:
        strategy: One of 'fixed', 'medium', or 'sentence'.

    Returns:
        A LangChain TextSplitter instance.

    Raises:
        ValueError: If an unknown strategy name is provided.
    """
    if strategy == "fixed":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.fixed_chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "medium":
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.medium_chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    elif strategy == "sentence":
        return NLTKTextSplitter(
            chunk_size=settings.fixed_chunk_size,
        )
    else:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Choose from: fixed, medium, sentence."
        )


# ---------------------------------------------------------------------------
# RAG Engine Class
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Workflow:
      1. Load a PDF document.
      2. Split it into chunks using a chosen strategy.
      3. Create embeddings and store them in a FAISS vector store.
      4. Answer questions by retrieving relevant chunks and passing
         them to an LLM for answer generation.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
        )
        self.vector_store: Optional[FAISS] = None
        self.qa_chain = None
        self.chunks = []
        self.current_strategy = None

        # LLM will be initialized lazily on first query
        self._llm = None

    # ----- Private helpers ------------------------------------------------

    def _get_llm(self):
        """Lazily initialize the HuggingFace LLM pipeline."""
        if self._llm is None:
            pipe = pipeline(
                "text2text-generation",
                model=settings.llm_model,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )
            self._llm = HuggingFacePipeline(pipeline=pipe)
        return self._llm

    def _build_qa_chain(self):
        """Build the RetrievalQA chain from the current vector store."""
        if self.vector_store is None:
            raise RuntimeError("No documents loaded. Upload a PDF first.")

        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Use the following context to answer the question. "
                "If you cannot find the answer in the context, say "
                "'I could not find the answer in the provided document.'\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self._get_llm(),
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": settings.top_k},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

    # ----- Public API -----------------------------------------------------

    def load_and_index(self, pdf_path: str, strategy: str = "fixed"):
        """
        Load a PDF, split it into chunks, and build a FAISS index.

        Args:
            pdf_path: Absolute path to the PDF file.
            strategy: Chunking strategy ('fixed', 'medium', or 'sentence').

        Returns:
            dict with keys 'num_chunks' and 'strategy'.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Step 1: Load PDF pages
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Step 2: Split into chunks
        splitter = get_text_splitter(strategy)
        self.chunks = splitter.split_documents(pages)
        self.current_strategy = strategy

        # Step 3: Build FAISS vector store
        self.vector_store = FAISS.from_documents(
            self.chunks, self.embeddings
        )

        # Step 4: Rebuild QA chain with new index
        self._build_qa_chain()

        return {
            "num_chunks": len(self.chunks),
            "strategy": strategy,
        }

    def query(self, question: str) -> dict:
        """
        Query the loaded document with a natural-language question.

        Args:
            question: The user's question string.

        Returns:
            dict with 'answer' and 'sources' (list of source chunk texts).
        """
        if self.qa_chain is None:
            raise RuntimeError("No documents loaded. Upload a PDF first.")

        result = self.qa_chain.invoke({"query": question})

        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "N/A"),
            })

        return {
            "answer": result.get("result", "No answer generated."),
            "sources": sources,
        }

    def retrieve_chunks(self, question: str, k: int = 4):
        """
        Retrieve the top-k most relevant chunks for a question
        (without generating an answer).

        Useful for evaluation and debugging.

        Args:
            question: The query string.
            k: Number of chunks to retrieve.

        Returns:
            List of Document objects.
        """
        if self.vector_store is None:
            raise RuntimeError("No documents loaded. Upload a PDF first.")

        return self.vector_store.similarity_search(question, k=k)
