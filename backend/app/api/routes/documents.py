import logging

from fastapi import APIRouter, Request

from app.api.schemas.documents import ChunkedDocumentResponse, ExtractedDocumentResponse
from app.services.document_chunking_service import DocumentChunkingService
from app.services.pdf_processing_service import PdfProcessingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get(
    "/{document_id}/text",
    response_model=ExtractedDocumentResponse,
    summary="Extract page-aware text from an uploaded PDF",
)
async def extract_document_text(
    document_id: str,
    request: Request,
) -> ExtractedDocumentResponse:
    processing_service: PdfProcessingService = request.app.state.pdf_processing_service
    extracted_document = await processing_service.extract_text(document_id)
    logger.info(
        "PDF text extracted successfully",
        extra={
            "document_id": extracted_document.document_id,
            "page_count": extracted_document.page_count,
            "total_characters": extracted_document.total_characters,
        },
    )
    return extracted_document


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkedDocumentResponse,
    summary="Chunk extracted PDF text with page references",
)
async def chunk_document(
    document_id: str,
    request: Request,
) -> ChunkedDocumentResponse:
    chunking_service: DocumentChunkingService = request.app.state.document_chunking_service
    chunked_document = await chunking_service.chunk_document(document_id)
    logger.info(
        "PDF text chunked successfully",
        extra={
            "document_id": chunked_document.document_id,
            "chunk_count": chunked_document.chunk_count,
            "chunk_size": chunked_document.chunk_size,
            "chunk_overlap": chunked_document.chunk_overlap,
        },
    )
    return chunked_document
