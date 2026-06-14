import logging

from fastapi import APIRouter, Request

from app.api.schemas.embeddings import (
    DocumentEmbeddingResponse,
    EmbeddingRequest,
    EmbeddingResponse,
)
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post(
    "",
    response_model=EmbeddingResponse,
    summary="Generate embeddings for text inputs",
)
async def create_embeddings(
    payload: EmbeddingRequest,
    request: Request,
) -> EmbeddingResponse:
    embedding_service: EmbeddingService = request.app.state.embedding_service
    result = await embedding_service.embed_texts(payload.inputs)
    logger.info(
        "Text embeddings generated successfully",
        extra={
            "input_count": result.input_count,
            "model": result.model,
            "total_tokens": result.usage.total_tokens,
        },
    )
    return result


@router.post(
    "/documents/{document_id}",
    response_model=DocumentEmbeddingResponse,
    summary="Generate embeddings for all chunks in an uploaded PDF",
)
async def create_document_embeddings(
    document_id: str,
    request: Request,
) -> DocumentEmbeddingResponse:
    embedding_service: EmbeddingService = request.app.state.embedding_service
    result = await embedding_service.embed_document_chunks(document_id)
    logger.info(
        "Document chunk embeddings generated successfully",
        extra={
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "model": result.model,
            "total_tokens": result.usage.total_tokens,
        },
    )
    return result
