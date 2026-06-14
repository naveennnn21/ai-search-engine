from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UploadedDocumentMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(..., description="Stable document identifier.")
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    uploaded_at: datetime
    relative_path: str


class UploadResponse(BaseModel):
    document: UploadedDocumentMetadata
