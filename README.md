# 🤖 RAG-Powered Knowledge Base Chatbot

> Ask questions about any PDF or website using **Retrieval-Augmented Generation (RAG)**  
> Built with LangChain · OpenAI · ChromaDB · Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 📐 Architecture

```
 ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
 │  PDFs /     │────▶│  LangChain   │────▶│   ChromaDB    │
 │  Websites   │     │  Chunker +   │     │ (Vector Store)│
 └─────────────┘     │  Embeddings  │     └──────┬────────┘
                     └──────────────┘            │ Similarity Search
                                                 ▼
 ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
 │   Answer +  │◀────│  GPT-4o-mini │◀────│  Top-K Chunks │
 │   Sources   │     │  (LLM)       │     │  (Context)    │
 └─────────────┘     └──────────────┘     └───────────────┘
```

**RAG Flow:**
1. Documents are split into chunks
2. Chunks are converted to vector embeddings
3. Vectors are stored in ChromaDB
4. On query: retrieve top-K most relevant chunks
5. Chunks + question are sent to GPT → grounded answer

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                    # Streamlit UI (main entry point)
├── src/
│   └── rag_pipeline.py       # Core RAG logic
├── tests/
│   └── test_rag_pipeline.py  # Unit tests
├── data/                     # Drop your PDFs here
├── .vscode/
│   ├── launch.json           # Run/debug configs
│   └── extensions.json       # Recommended extensions
├── .github/
│   └── workflows/ci.yml      # GitHub Actions CI
├── .env.example              # Environment variables template
├── .gitignore
└── requirements.txt
```

---

## 🚀 Step-by-Step Setup Guide

### ─── PHASE 1: Environment Setup ───────────────────────

#### Step 1 — Install Prerequisites

Make sure you have these installed:
- **Python 3.11+** → https://www.python.org/downloads/
- **Git** → https://git-scm.com
- **VSCode** → https://code.visualstudio.com

Verify installations:
```bash
python --version    # Should show 3.11+
git --version
code --version
```

---

#### Step 2 — Create GitHub Repository

1. Go to **https://github.com/new**
2. Repository name: `rag-knowledge-chatbot`
3. Set to **Public** (for portfolio visibility)
4. Check **Add a README file**
5. Click **Create repository**

Then clone it locally:
```bash
git clone https://github.com/YOUR_USERNAME/rag-knowledge-chatbot.git
cd rag-knowledge-chatbot
```

---

#### Step 3 — Open in VSCode

```bash
code .
```

VSCode will open. When prompted:
- Install recommended extensions (Python, Pylint, Black Formatter)
- Trust the workspace

---

#### Step 4 — Create Virtual Environment

In the VSCode terminal (`Ctrl + `` ` ``):
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Confirm activation (should show venv in prompt)
which python
```

---

#### Step 5 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: LangChain, OpenAI, ChromaDB, Streamlit, PyPDF, and more.

---

#### Step 6 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

> 💡 Get your API key at https://platform.openai.com/api-keys  
> ⚠️ Never commit your `.env` file — it's already in `.gitignore`

---

### ─── PHASE 2: Understanding the Code ──────────────────

Open `src/rag_pipeline.py` in VSCode and study each step:

| Step | Function | What it does |
|------|----------|-------------|
| 1 | `load_pdfs()` | Reads PDF files and extracts text |
| 2 | `split_documents()` | Breaks text into overlapping chunks |
| 3 | `create_vectorstore()` | Converts chunks to vectors via OpenAI Embeddings, stores in ChromaDB |
| 4 | `build_rag_chain()` | Connects retriever + GPT into a QA chain |
| 5 | `ask()` | Queries the chain, returns answer + sources |

**Key concept — chunk_overlap:**
Overlapping chunks ensure context doesn't get cut off at chunk boundaries.

```
Chunk 1: "...The policy states that employees..."
                                    ↑ overlap
Chunk 2:                "...employees are entitled to 20 days..."
```

---

### ─── PHASE 3: Run the Application ─────────────────────

#### Step 7 — Add Your Documents

Drop any PDF files into the `data/` folder:
```bash
# Example: copy a PDF
cp ~/Downloads/your-document.pdf data/
```

Or use the UI to upload PDFs directly.

---

#### Step 8 — Launch the App

```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501**

**Using the app:**
1. Enter your OpenAI API key in the sidebar (or set it in `.env`)
2. Upload PDFs **or** enter a URL
3. Click **⚡ Build Knowledge Base**
4. Start asking questions in the chat!

---

#### Step 9 — Debug in VSCode (Optional)

Use the built-in launch configs:
1. Open **Run & Debug** panel (`Ctrl+Shift+D`)
2. Select **"Run Streamlit App"** from dropdown
3. Press **F5** to start with debugger attached
4. Set breakpoints anywhere in `rag_pipeline.py` to inspect variables

---

### ─── PHASE 4: Testing ──────────────────────────────────

#### Step 10 — Run Unit Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_rag_pipeline.py::test_split_documents_basic PASSED
tests/test_rag_pipeline.py::test_split_documents_overlap PASSED
tests/test_rag_pipeline.py::test_split_documents_preserves_metadata PASSED
tests/test_rag_pipeline.py::test_ask_returns_answer_and_sources PASSED
tests/test_rag_pipeline.py::test_ask_deduplicates_sources PASSED
```

---

### ─── PHASE 5: Push to GitHub ───────────────────────────

#### Step 11 — Commit and Push

```bash
# Stage all files
git add .

# First commit
git commit -m "feat: initial RAG chatbot implementation"

# Push to GitHub
git push origin main
```

---

#### Step 12 — Add GitHub Actions Secret

So CI can run tests with a real API key:
1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `OPENAI_API_KEY`
4. Value: your API key
5. Click **Add secret**

Now every push triggers automated tests! ✅

---

### ─── PHASE 6: Improvements to Showcase ────────────────

Once the base project works, add these enhancements to stand out:

#### 🔧 Enhancement 1 — Switch to Pinecone (Cloud Vector DB)
```python
from langchain_pinecone import PineconeVectorStore
# Replace ChromaDB with Pinecone for production scalability
```

#### 🔧 Enhancement 2 — Add Conversation Memory
```python
from langchain.memory import ConversationBufferWindowMemory
# Lets the chatbot remember previous questions in a session
```

#### 🔧 Enhancement 3 — Add Evaluation (RAGAS)
```bash
pip install ragas
# Measure: Faithfulness, Answer Relevancy, Context Recall
```

#### 🔧 Enhancement 4 — Gemini as Alternative LLM
```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
```

#### 🔧 Enhancement 5 — Deploy to Streamlit Cloud (Free!)
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo → Deploy!
4. Add `OPENAI_API_KEY` in app secrets

---

## 🎯 What This Project Demonstrates (For Recruiters)

| Skill | How it's shown |
|-------|---------------|
| RAG Architecture | Full pipeline: ingest → chunk → embed → retrieve → generate |
| LangChain | Loaders, splitters, chains, retrievers, prompts |
| Vector Databases | ChromaDB with MMR retrieval strategy |
| LLM Integration | OpenAI GPT with custom prompt templates |
| MLOps Basics | CI/CD with GitHub Actions, environment management |
| Python Best Practices | Modular code, unit tests, type hints, dotenv |
| UI Development | Streamlit with session state management |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini / GPT-4o |
| Embeddings | OpenAI text-embedding-3-small |
| Framework | LangChain 0.3 |
| Vector DB | ChromaDB (local) / Pinecone (cloud) |
| UI | Streamlit |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |

---

## 📝 License

MIT — free to use for portfolio and learning.
