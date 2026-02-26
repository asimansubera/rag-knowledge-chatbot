"""
RAG Chatbot - Streamlit UI
Run: streamlit run app.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from src.rag_pipeline import (
    load_pdfs, load_website, split_documents,
    create_vectorstore, load_vectorstore, build_rag_chain, ask
)

load_dotenv()

# ─── Page Config ───────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Base Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG-Powered Knowledge Base Chatbot")
st.caption("Upload PDFs or enter a URL to create your knowledge base, then ask questions!")

# ─── Session State ─────────────────────────────
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb_loaded" not in st.session_state:
    st.session_state.kb_loaded = False

# ─── Sidebar: Knowledge Base Setup ─────────────
with st.sidebar:
    st.header("📚 Knowledge Base Setup")

    # API Key input
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    # Source selection
    source_type = st.radio("Choose Document Source", ["📄 Upload PDFs", "🌐 Website URL"])

    docs = []

    if source_type == "📄 Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            with st.spinner("Loading PDFs..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    for f in uploaded_files:
                        path = Path(tmpdir) / f.name
                        path.write_bytes(f.read())
                    docs = load_pdfs(tmpdir)

    else:
        url = st.text_input("Enter Website URL", placeholder="https://example.com/docs")
        if url:
            with st.spinner("Scraping website..."):
                try:
                    docs = load_website(url)
                except Exception as e:
                    st.error(f"Failed to load URL: {e}")

    # Model selection
    st.divider()
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 200, 50)

    # Build KB button
    if st.button("⚡ Build Knowledge Base", type="primary", disabled=len(docs) == 0):
        with st.spinner("Chunking & embedding documents..."):
            try:
                chunks = split_documents(docs, chunk_size, chunk_overlap)
                vs = create_vectorstore(chunks)
                st.session_state.chain = build_rag_chain(vs, model)
                st.session_state.kb_loaded = True
                st.session_state.messages = []
                st.success(f"✅ Knowledge base ready! ({len(chunks)} chunks indexed)")
            except Exception as e:
                st.error(f"Error building KB: {e}")

    if st.session_state.kb_loaded:
        st.success("🟢 Knowledge Base Active")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ─── Chat Interface ────────────────────────────
if not st.session_state.kb_loaded:
    st.info("👈 Set up your knowledge base in the sidebar to start chatting!")

    with st.expander("💡 How it works"):
        st.markdown("""
        1. **Upload PDFs** or enter a website URL
        2. Click **Build Knowledge Base** — documents are chunked & embedded
        3. **Ask questions** — the RAG pipeline retrieves relevant context and answers using GPT
        4. **Sources** are shown with every answer for transparency
        """)
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for s in msg["sources"]:
                        st.caption(f"• {s}")

    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask(st.session_state.chain, question)
                st.write(response["answer"])
                if response["sources"]:
                    with st.expander("📄 Sources"):
                        for s in response["sources"]:
                            st.caption(f"• {s}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"]
        })
