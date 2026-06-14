import logging

from fastapi import APIRouter, Request

from app.api.schemas.vector_store import (
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStoreLoadResponse,
    VectorStoreSaveResponse,
    VectorStoreStatsResponse,
    VectorUpsertResponse,
)
from app.services.vector_store_service import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector-store", tags=["vector-store"])


@router.post(
    "/documents/{document_id}",
    response_model=VectorUpsertResponse,
    summary="Embed and insert document chunks into FAISS",
)
async def index_document(
    document_id: str,
    request: Request,
) -> VectorUpsertResponse:
    vector_store_service: VectorStoreService = request.app.state.vector_store_service
    result = await vector_store_service.index_document(document_id)
    logger.info(
        "Document vectors indexed successfully",
        extra={
            "document_id": result.document_id,
            "inserted_count": result.inserted_count,
            "total_vectors": result.total_vectors,
        },
    )
    return result


@router.post(
    "/search",
    response_model=VectorSearchResponse,
    summary="Search FAISS with an embedded query",
)
async def search_vectors(
    payload: VectorSearchRequest,
    request: Request,
) -> VectorSearchResponse:
    vector_store_service: VectorStoreService = request.app.state.vector_store_service
    result = await vector_store_service.search_text(payload.query, payload.top_k)
    logger.info(
        "Vector search completed successfully",
        extra={
            "top_k": payload.top_k,
            "match_count": len(result.matches),
            "total_vectors": result.total_vectors,
        },
    )
    return result


@router.post(
    "/save",
    response_model=VectorStoreSaveResponse,
    summary="Persist the FAISS index and metadata",
)
async def save_vector_store(request: Request) -> VectorStoreSaveResponse:
    vector_store_service: VectorStoreService = request.app.state.vector_store_service
    return await vector_store_service.save()


@router.post(
    "/load",
    response_model=VectorStoreLoadResponse,
    summary="Load the FAISS index and metadata from disk",
)
async def load_vector_store(request: Request) -> VectorStoreLoadResponse:
    vector_store_service: VectorStoreService = request.app.state.vector_store_service
    return await vector_store_service.load()


@router.get(
    "/stats",
    response_model=VectorStoreStatsResponse,
    summary="Inspect the loaded FAISS index",
)
async def vector_store_stats(request: Request) -> VectorStoreStatsResponse:
    vector_store_service: VectorStoreService = request.app.state.vector_store_service
    return vector_store_service.stats()
