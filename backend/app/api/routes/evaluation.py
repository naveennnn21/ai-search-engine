from fastapi import APIRouter, Request

from app.api.schemas.evaluation import EvaluationRequest, EvaluationResponse
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post(
    "/ragas",
    response_model=EvaluationResponse,
    summary="Evaluate RAG outputs with RAGAS",
)
async def evaluate_with_ragas(
    payload: EvaluationRequest,
    request: Request,
) -> EvaluationResponse:
    evaluation_service: EvaluationService = request.app.state.evaluation_service
    return await evaluation_service.evaluate(payload.examples)
