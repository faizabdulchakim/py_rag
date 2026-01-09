# Local RAG API with FastAPI, Ollama, and LangChain

This project is a **manual Retrieval-Augmented Generation (RAG)** implementation using:
- **FastAPI** as the API server
- **Ollama** for local LLM and embeddings
- **LangChain** (manual pipeline, no RetrievalQA)
- **ChromaDB** as vector store

The pipeline includes:
- Document ingestion
- Text chunking
- Embedding
- Similarity retrieval
- Manual prompt assembly
- Token usage estimation

---

## 📁 Project Structure

py_rag/
│
├── data/
│ └── docs.txt # Source documents
│
├── chroma_db/ # Persisted vector database
│
├── venv/ # Python virtual environment
│
├── main.py # FastAPI RAG application
├── README.md
└── .gitignore


---

## 🔧 Requirements

- Python **3.10 – 3.12**
- Ollama installed and running
- Models pulled:
  - `phi4:latest`
  - `nomic-embed-text`

---

## 🦙 Install Ollama Models

```bash
ollama pull phi4:latest
ollama pull nomic-embed-text

Verify:

ollama list


🐍 Setup Python Environment
1️⃣ Create virtual environment
python -m venv venv

2️⃣ Activate virtual environment

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

3️⃣ Install dependencies
pip install fastapi uvicorn langchain langchain-community langchain-ollama chromadb tiktoken

📄 Prepare Documents

Put your knowledge content inside:

data/docs.txt


Example:

FastAPI is a modern Python web framework for building APIs.
Ollama allows running large language models locally.
RAG combines retrieval and generation for better answers.

▶️ Run the Server
uvicorn main:app --reload


Server will run at:

http://127.0.0.1:8000


Swagger UI:

http://127.0.0.1:8000/docs

🔌 API Endpoint
POST /rag

Request

{
  "question": "What is RAG?",
  "top_k": 3
}


Response

{
  "question": "...",
  "answer": "...",
  "retrieved_documents": [...],
  "prompt_used": "...",
  "token_usage_estimate": {
    "prompt_tokens": 123,
    "completion_tokens": 45,
    "total_tokens": 168
  }
}
