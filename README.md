# AI Search Engine

Production-oriented AI search engine with FastAPI, FAISS, OpenAI embeddings, hybrid retrieval, RAG, RAGAS evaluation, and a React TypeScript frontend.

## Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="your_api_key"
uvicorn app.main:app --reload
```

Backend API: http://127.0.0.1:8000/docs

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend app: http://localhost:5173

## Workflow

1. Upload a PDF.
2. Index it for search.
3. Search documents with hybrid BM25 + vector retrieval.
4. Chat with indexed documents and inspect citations.
