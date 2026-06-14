import logging

from fastapi import APIRouter, Request

from app.api.schemas.retrieval import QueryExpansionResponse, RetrievalRequest, RetrievalResponse
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    summary="Retrieve top matching chunks with hybrid search",
)
async def retrieve_chunks(
    payload: RetrievalRequest,
    request: Request,
) -> RetrievalResponse:
    retrieval_service: RetrievalService = request.app.state.retrieval_service
    result = await retrieval_service.retrieve(
        query=payload.query,
        top_k=payload.top_k,
        use_query_expansion=payload.use_query_expansion,
        use_reranking=payload.use_reranking,
    )
    logger.info(
        "Retrieval completed",
        extra={
            "top_k": payload.top_k,
            "matches": len(result.matches),
            "candidates": result.total_candidates,
        },
    )
    return result


@router.post(
    "/expand",
    response_model=QueryExpansionResponse,
    summary="Expand a search query",
)
async def expand_query(
    payload: RetrievalRequest,
    request: Request,
) -> QueryExpansionResponse:
    retrieval_service: RetrievalService = request.app.state.retrieval_service
    expanded_queries = await retrieval_service.expand_query(payload.query)
    return QueryExpansionResponse(query=payload.query, expanded_queries=expanded_queries)
