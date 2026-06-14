from pydantic import BaseModel, Field

from app.api.schemas.retrieval import RetrievalMatch


class Citation(BaseModel):
    source_id: str
    document_id: str
    chunk_id: str
    page_numbers: list[int]
    snippet: str


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=20)


class RagResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalMatch]
