from pydantic import BaseModel, Field

from app.api.schemas.documents import PageReference
from app.api.schemas.embeddings import EmbeddingUsage


class VectorMetadata(BaseModel):
    vector_id: int = Field(..., ge=0)
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    page_numbers: list[int]
    page_references: list[PageReference]
    metadata: dict[str, str | int]


class VectorUpsertResponse(BaseModel):
    document_id: str
    filename: str
    inserted_count: int = Field(..., ge=0)
    skipped_count: int = Field(..., ge=0)
    total_vectors: int = Field(..., ge=0)
    index_path: str
    metadata_path: str
    usage: EmbeddingUsage


class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class VectorSearchMatch(BaseModel):
    rank: int = Field(..., ge=1)
    score: float
    vector_id: int = Field(..., ge=0)
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    page_numbers: list[int]
    page_references: list[PageReference]
    metadata: dict[str, str | int]


class VectorSearchResponse(BaseModel):
    query: str
    model: str
    top_k: int = Field(..., ge=1)
    total_vectors: int = Field(..., ge=0)
    matches: list[VectorSearchMatch]
    usage: EmbeddingUsage


class VectorStoreSaveResponse(BaseModel):
    saved: bool
    total_vectors: int = Field(..., ge=0)
    index_path: str
    metadata_path: str


class VectorStoreLoadResponse(BaseModel):
    loaded: bool
    total_vectors: int = Field(..., ge=0)
    index_path: str
    metadata_path: str


class VectorStoreStatsResponse(BaseModel):
    dimension: int | None
    total_vectors: int = Field(..., ge=0)
    index_path: str
    metadata_path: str
