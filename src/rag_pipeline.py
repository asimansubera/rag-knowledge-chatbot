"""
RAG Pipeline - Core logic for document ingestion and retrieval
"""

import os
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# STEP 1: Load Documents
# ──────────────────────────────────────────────

def load_pdfs(pdf_dir: str = "data/") -> List:
    """Load all PDFs from a directory."""
    loader = DirectoryLoader(pdf_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} PDF pages from '{pdf_dir}'")
    return docs


def load_website(url: str) -> List:
    """Load and parse a webpage."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"✅ Loaded webpage: {url}")
    return docs


# ──────────────────────────────────────────────
# STEP 2: Chunk Documents
# ──────────────────────────────────────────────

def split_documents(docs: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ──────────────────────────────────────────────
# STEP 3: Embed + Store in Vector DB
# ──────────────────────────────────────────────

def create_vectorstore(chunks: List, persist_dir: str = "chroma_db") -> Chroma:
    """Create and persist a ChromaDB vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Vector store created and saved to '{persist_dir}'")
    return vectorstore


def load_vectorstore(persist_dir: str = "chroma_db") -> Chroma:
    """Load an existing ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print(f"✅ Vector store loaded from '{persist_dir}'")
    return vectorstore


# ──────────────────────────────────────────────
# STEP 4: Build RAG Chain
# ──────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def build_rag_chain(vectorstore: Chroma, model: str = "gpt-4o-mini") -> RetrievalQA:
    """Build the RAG chain with retriever + LLM."""
    llm = ChatOpenAI(model=model, temperature=0)
    retriever = vectorstore.as_retriever(
        search_type="mmr",           # Max Marginal Relevance for diversity
        search_kwargs={"k": 5, "fetch_k": 20}
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )
    print("✅ RAG chain ready!")
    return chain


# ──────────────────────────────────────────────
# STEP 5: Query
# ──────────────────────────────────────────────

def ask(chain: RetrievalQA, question: str) -> dict:
    """Ask a question and return answer + sources."""
    result = chain.invoke({"query": question})
    sources = list({doc.metadata.get("source", "Unknown") for doc in result["source_documents"]})
    return {
        "answer": result["result"],
        "sources": sources
    }


# ──────────────────────────────────────────────
# Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Load & process documents
    docs = load_pdfs("data/")
    chunks = split_documents(docs)
    vs = create_vectorstore(chunks)
    chain = build_rag_chain(vs)

    # Ask a question
    response = ask(chain, "What is this document about?")
    print("\n📌 Answer:", response["answer"])
    print("📄 Sources:", response["sources"])
