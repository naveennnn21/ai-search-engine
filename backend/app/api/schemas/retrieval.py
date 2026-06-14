from pydantic import BaseModel, Field

from app.api.schemas.documents import PageReference
from app.api.schemas.embeddings import EmbeddingUsage


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    use_query_expansion: bool = True
    use_reranking: bool = True


class RetrievalMatch(BaseModel):
    rank: int = Field(..., ge=1)
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    similarity_score: float
    vector_score: float
    bm25_score: float
    rerank_score: float | None = None
    page_numbers: list[int]
    page_references: list[PageReference]
    metadata: dict[str, str | int]


class RetrievalResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    top_k: int = Field(..., ge=1)
    total_candidates: int = Field(..., ge=0)
    matches: list[RetrievalMatch]
    usage: EmbeddingUsage


class QueryExpansionResponse(BaseModel):
    query: str
    expanded_queries: list[str]
