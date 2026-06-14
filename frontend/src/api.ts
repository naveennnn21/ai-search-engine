import type {
  RagResponse,
  RetrievalResponse,
  UploadResponse,
  VectorUpsertResponse
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/api/v1";

async function parseResponse<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.error?.message ?? "Request failed";
    throw new Error(message);
  }
  return payload as T;
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData
  });
  return parseResponse<UploadResponse>(response);
}

export async function indexDocument(documentId: string): Promise<VectorUpsertResponse> {
  const response = await fetch(`${API_BASE_URL}/vector-store/documents/${documentId}`, {
    method: "POST"
  });
  return parseResponse<VectorUpsertResponse>(response);
}

export async function searchDocuments(query: string): Promise<RetrievalResponse> {
  const response = await fetch(`${API_BASE_URL}/retrieval/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: 10,
      use_query_expansion: true,
      use_reranking: true
    })
  });
  return parseResponse<RetrievalResponse>(response);
}

export async function chatWithDocuments(query: string): Promise<RagResponse> {
  const response = await fetch(`${API_BASE_URL}/rag/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 10 })
  });
  return parseResponse<RagResponse>(response);
}
