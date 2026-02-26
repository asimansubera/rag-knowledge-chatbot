"""
Tests for RAG pipeline components
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from src.rag_pipeline import split_documents


# ─── Test: Document Splitting ─────────────────

def test_split_documents_basic():
    """Splitting docs should produce more chunks than original pages."""
    from langchain.schema import Document
    docs = [Document(page_content="A" * 3000, metadata={"source": "test.pdf"})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_split_documents_overlap():
    """Chunks should respect overlap settings."""
    from langchain.schema import Document
    docs = [Document(page_content="Hello world. " * 200, metadata={"source": "test.pdf"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=50)
    assert all(len(c.page_content) <= 250 for c in chunks)


def test_split_documents_preserves_metadata():
    """Metadata should be preserved after splitting."""
    from langchain.schema import Document
    docs = [Document(page_content="X" * 2000, metadata={"source": "my_doc.pdf", "page": 1})]
    chunks = split_documents(docs)
    assert all(c.metadata.get("source") == "my_doc.pdf" for c in chunks)


# ─── Test: Ask function ───────────────────────

def test_ask_returns_answer_and_sources():
    """ask() should return a dict with answer and sources keys."""
    mock_chain = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "test.pdf"}
    mock_chain.invoke.return_value = {
        "result": "This is the answer.",
        "source_documents": [mock_doc]
    }

    from src.rag_pipeline import ask
    result = ask(mock_chain, "What is this about?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "This is the answer."
    assert "test.pdf" in result["sources"]


def test_ask_deduplicates_sources():
    """Sources should be deduplicated."""
    mock_chain = MagicMock()
    mock_doc1 = MagicMock()
    mock_doc1.metadata = {"source": "doc.pdf"}
    mock_doc2 = MagicMock()
    mock_doc2.metadata = {"source": "doc.pdf"}  # same source
    mock_chain.invoke.return_value = {
        "result": "Answer",
        "source_documents": [mock_doc1, mock_doc2]
    }

    from src.rag_pipeline import ask
    result = ask(mock_chain, "question")
    assert len(result["sources"]) == 1
