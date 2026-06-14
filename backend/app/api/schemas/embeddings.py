from pydantic import BaseModel, Field

from app.api.schemas.documents import PageReference


class EmbeddingRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1, max_length=2048)


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class TextEmbedding(BaseModel):
    index: int = Field(..., ge=0)
    text: str
    embedding: list[float]
    dimensions: int = Field(..., ge=1)


class EmbeddingResponse(BaseModel):
    model: str
    input_count: int = Field(..., ge=0)
    embeddings: list[TextEmbedding]
    usage: EmbeddingUsage


class ChunkEmbedding(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    embedding: list[float]
    dimensions: int = Field(..., ge=1)
    page_numbers: list[int]
    page_references: list[PageReference]
    metadata: dict[str, str | int]


class DocumentEmbeddingResponse(BaseModel):
    document_id: str
    filename: str
    model: str
    chunk_count: int = Field(..., ge=0)
    embeddings: list[ChunkEmbedding]
    usage: EmbeddingUsage
