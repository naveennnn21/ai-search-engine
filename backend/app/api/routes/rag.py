import logging

from fastapi import APIRouter, Request

from app.api.schemas.rag import RagRequest, RagResponse
from app.services.rag_service import RagService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/chat",
    response_model=RagResponse,
    summary="Answer a question using retrieved document chunks",
)
async def chat_with_documents(
    payload: RagRequest,
    request: Request,
) -> RagResponse:
    rag_service: RagService = request.app.state.rag_service
    result = await rag_service.answer(payload.query, payload.top_k)
    logger.info(
        "RAG answer generated",
        extra={"citations": len(result.citations), "retrieved_chunks": len(result.retrieved_chunks)},
    )
    return result
