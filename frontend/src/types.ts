export type UploadedDocument = {
  document_id: string;
  original_filename: string;
  stored_filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  uploaded_at: string;
  relative_path: string;
};

export type UploadResponse = {
  document: UploadedDocument;
};

export type PageReference = {
  page_number: number;
  start_character: number;
  end_character: number;
};

export type RetrievalMatch = {
  rank: number;
  chunk_id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  similarity_score: number;
  vector_score: number;
  bm25_score: number;
  rerank_score: number | null;
  page_numbers: number[];
  page_references: PageReference[];
  metadata: Record<string, string | number>;
};

export type RetrievalResponse = {
  query: string;
  expanded_queries: string[];
  top_k: number;
  total_candidates: number;
  matches: RetrievalMatch[];
};

export type Citation = {
  source_id: string;
  document_id: string;
  chunk_id: string;
  page_numbers: number[];
  snippet: string;
};

export type RagResponse = {
  query: string;
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievalMatch[];
};

export type VectorUpsertResponse = {
  document_id: string;
  filename: string;
  inserted_count: number;
  skipped_count: number;
  total_vectors: number;
};
