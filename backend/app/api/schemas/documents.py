from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    page_number: int = Field(..., ge=1)
    text: str
    character_count: int = Field(..., ge=0)


class ExtractedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int = Field(..., ge=0)
    total_characters: int = Field(..., ge=0)
    pages: list[ExtractedPage]


class PageReference(BaseModel):
    page_number: int = Field(..., ge=1)
    start_character: int = Field(..., ge=0)
    end_character: int = Field(..., ge=0)


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int = Field(..., ge=0)
    text: str
    character_count: int = Field(..., ge=0)
    start_character: int = Field(..., ge=0)
    end_character: int = Field(..., ge=0)
    page_numbers: list[int]
    page_references: list[PageReference]
    metadata: dict[str, str | int]


class ChunkedDocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunk_size: int = Field(..., ge=1)
    chunk_overlap: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    chunks: list[DocumentChunk]
