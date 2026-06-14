# AI Search Engine Backend

Module 1 implements a FastAPI PDF upload service.
Module 2 adds page-aware PDF text extraction.
Module 3 adds document chunking with page references.
Module 4 adds async OpenAI embedding generation.
Module 5 adds FAISS vector indexing, persistence, and search.
Module 6 adds hybrid retrieval, RAG answers, and RAGAS evaluation.

## Run locally

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Upload a PDF

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@C:\path\to\document.pdf;type=application/pdf"
```

## Extract text from an uploaded PDF

Use the `document_id` returned from the upload response:

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents/{document_id}/text"
```

## Chunk an uploaded PDF

Use the `document_id` returned from the upload response:

```powershell
curl.exe "http://127.0.0.1:8000/api/v1/documents/{document_id}/chunks"
```

## Generate embeddings for text

Set `OPENAI_API_KEY` in your environment first.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/embeddings" `
  -H "Content-Type: application/json" `
  -d "{\"inputs\":[\"semantic search text\",\"another text\"]}"
```

## Generate embeddings for document chunks

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/embeddings/documents/{document_id}"
```

## Insert document vectors into FAISS

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/vector-store/documents/{document_id}"
```

## Search vectors

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/vector-store/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"what does the document say about revenue?\",\"top_k\":5}"
```

## Save or load the vector store

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/vector-store/save"
curl.exe -X POST "http://127.0.0.1:8000/api/v1/vector-store/load"
curl.exe "http://127.0.0.1:8000/api/v1/vector-store/stats"
```

## Hybrid retrieval

Returns the top 10 matching chunks with similarity scores.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/retrieval/search" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"what are the payment terms?\",\"top_k\":10}"
```

## RAG chat

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/rag/chat" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Summarize the payment obligations\",\"top_k\":10}"
```

## RAGAS evaluation

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/evaluation/ragas" `
  -H "Content-Type: application/json" `
  -d "{\"examples\":[{\"question\":\"What is covered?\",\"answer\":\"...\",\"contexts\":[\"...\"],\"ground_truth\":\"...\"}]}"
```
