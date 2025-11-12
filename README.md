# 🧠 RAG Document QA System (FastAPI + OpenAI + FAISS + Ollama)

A **Retrieval-Augmented Generation (RAG)** API built with **FastAPI**, combining:

- **OpenAI embeddings** (for retrieval)
- **FAISS** (for fast vector search)
- **Ollama** (for local LLM answer generation)
- **Recursive chunking** via LangChain
- **Modular service architecture**
```

## 🏗️ Project Structure
app/
├─ main.py
├─ api/
│ ├─ upload.py # Upload & process documents
│ ├─ query.py # Query documents
│ └─ documents.py # List / delete indexed docs
├─ services/
│ ├─ file_readers.py # PDF/TXT extraction
│ ├─ docs_service.py # Chunking + embedding
│ ├─ embedding_service.py
│ ├─ vector_service.py # FAISS vector storage
│ └─ llm_service.py # Ollama LLM integration
├─ core/
│ ├─ config.py # Env vars & constants
│ └─ logging_config.py # Logging setup
└─ experiments/
└─ compare_chunk_sizes.py


---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone git@github.com:Titanflame26/Assignment-3.git


2️⃣ Create a virtual environment

python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)

3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Create .env file
# --- OpenAI ---
OPENAI_API_KEY=sk-xxxxxxx
EMBEDDING_MODEL=text-embedding-3-small

# --- Ollama (local LLM) ---
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434

# --- FAISS / Storage ---
DATA_DIR=./data/index
TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# --- App Settings ---
ENVIRONMENT=development
LOG_LEVEL=INFO

🧩 Running Ollama
ollama serve
ollama pull llama3

🚀 Run the FastAPI App

Start the API server:

uvicorn app.main:app --reload

Swagger UI → http://127.0.0.1:8000/docs

Health Check → http://127.0.0.1:8000/
