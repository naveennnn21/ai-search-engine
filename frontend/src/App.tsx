import { useState } from "react";
import { FileUp, MessageSquareText, Search, Server, UploadCloud } from "lucide-react";
import { chatWithDocuments, indexDocument, searchDocuments, uploadPdf } from "./api";
import type { RagResponse, RetrievalResponse, UploadedDocument, VectorUpsertResponse } from "./types";

type Page = "upload" | "search" | "chat";

export function App() {
  const [page, setPage] = useState<Page>("upload");
  const [uploadedDocument, setUploadedDocument] = useState<UploadedDocument | null>(null);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Server size={22} />
          <div>
            <strong>AI Search Engine</strong>
            <span>RAG workspace</span>
          </div>
        </div>
        <nav className="nav">
          <button className={page === "upload" ? "active" : ""} onClick={() => setPage("upload")}>
            <FileUp size={18} /> Upload PDF
          </button>
          <button className={page === "search" ? "active" : ""} onClick={() => setPage("search")}>
            <Search size={18} /> Search Documents
          </button>
          <button className={page === "chat" ? "active" : ""} onClick={() => setPage("chat")}>
            <MessageSquareText size={18} /> Chat Interface
          </button>
        </nav>
      </aside>

      <main className="workspace">
        {page === "upload" && (
          <UploadPage uploadedDocument={uploadedDocument} onUploaded={setUploadedDocument} />
        )}
        {page === "search" && <SearchPage />}
        {page === "chat" && <ChatPage />}
      </main>
    </div>
  );
}

function UploadPage({
  uploadedDocument,
  onUploaded
}: {
  uploadedDocument: UploadedDocument | null;
  onUploaded: (document: UploadedDocument) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<string>("");
  const [isBusy, setIsBusy] = useState(false);
  const [indexResult, setIndexResult] = useState<VectorUpsertResponse | null>(null);

  async function handleUpload() {
    if (!file) return;
    setIsBusy(true);
    setStatus("Uploading PDF");
    try {
      const result = await uploadPdf(file);
      onUploaded(result.document);
      setStatus("Upload complete");
      setIndexResult(null);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsBusy(false);
    }
  }

  async function handleIndex() {
    if (!uploadedDocument) return;
    setIsBusy(true);
    setStatus("Embedding chunks and updating FAISS");
    try {
      const result = await indexDocument(uploadedDocument.document_id);
      setIndexResult(result);
      setStatus("Document indexed");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Indexing failed");
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h1>Upload PDF</h1>
        <p>Upload a source document and index it for hybrid retrieval.</p>
      </div>
      <div className="upload-box">
        <UploadCloud size={36} />
        <input
          type="file"
          accept="application/pdf"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <button disabled={!file || isBusy} onClick={handleUpload}>Upload</button>
      </div>
      {uploadedDocument && (
        <div className="status-grid">
          <div><span>Document ID</span><strong>{uploadedDocument.document_id}</strong></div>
          <div><span>Filename</span><strong>{uploadedDocument.original_filename}</strong></div>
          <div><span>Size</span><strong>{Math.round(uploadedDocument.size_bytes / 1024)} KB</strong></div>
          <button disabled={isBusy} onClick={handleIndex}>Index for Search</button>
        </div>
      )}
      {indexResult && (
        <div className="notice">
          Inserted {indexResult.inserted_count} chunks, skipped {indexResult.skipped_count}. Total vectors: {indexResult.total_vectors}.
        </div>
      )}
      {status && <div className="notice subtle">{status}</div>}
    </section>
  );
}

function SearchPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<RetrievalResponse | null>(null);
  const [error, setError] = useState("");
  const [isBusy, setIsBusy] = useState(false);

  async function handleSearch() {
    if (!query.trim()) return;
    setIsBusy(true);
    setError("");
    try {
      setResult(await searchDocuments(query));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h1>Search Documents</h1>
        <p>Hybrid BM25 and vector retrieval with query expansion and reranking.</p>
      </div>
      <div className="search-row">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask for matching chunks" />
        <button disabled={isBusy || !query.trim()} onClick={handleSearch}>Search</button>
      </div>
      {error && <div className="notice danger">{error}</div>}
      {result && (
        <>
          <div className="query-list">
            {result.expanded_queries.map((item) => <span key={item}>{item}</span>)}
          </div>
          <div className="results">
            {result.matches.map((match) => (
              <article className="result-item" key={match.chunk_id}>
                <div className="result-meta">
                  <strong>#{match.rank} Score {match.similarity_score.toFixed(3)}</strong>
                  <span>Pages {match.page_numbers.join(", ") || "n/a"}</span>
                </div>
                <p>{match.text}</p>
                <footer>Vector {match.vector_score.toFixed(3)} / BM25 {match.bm25_score.toFixed(3)}</footer>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function ChatPage() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [lastResponse, setLastResponse] = useState<RagResponse | null>(null);
  const [error, setError] = useState("");
  const [isBusy, setIsBusy] = useState(false);

  async function handleSend() {
    if (!query.trim()) return;
    const userQuery = query.trim();
    setMessages((current) => [...current, { role: "user", content: userQuery }]);
    setQuery("");
    setIsBusy(true);
    setError("");
    try {
      const response = await chatWithDocuments(userQuery);
      setLastResponse(response);
      setMessages((current) => [...current, { role: "assistant", content: response.answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Chat failed");
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section className="panel chat-panel">
      <div className="panel-header">
        <h1>Chat Interface</h1>
        <p>Answers are generated from retrieved chunks and returned with citations.</p>
      </div>
      <div className="chat-log">
        {messages.map((message, index) => (
          <div className={`message ${message.role}`} key={`${message.role}-${index}`}>{message.content}</div>
        ))}
      </div>
      {lastResponse && (
        <div className="citations">
          {lastResponse.citations.map((citation) => (
            <div key={citation.source_id}>
              <strong>{citation.source_id}</strong>
              <span>Pages {citation.page_numbers.join(", ") || "n/a"}</span>
              <p>{citation.snippet}</p>
            </div>
          ))}
        </div>
      )}
      {error && <div className="notice danger">{error}</div>}
      <div className="search-row">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question about indexed PDFs" />
        <button disabled={isBusy || !query.trim()} onClick={handleSend}>Send</button>
      </div>
    </section>
  );
}
