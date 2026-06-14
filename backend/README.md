# AI Search Engine Backend

Module 1 implements a FastAPI PDF upload service.

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
