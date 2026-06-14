from typing import Any

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(..., min_length=1)
    ground_truth: str | None = None


class EvaluationRequest(BaseModel):
    examples: list[EvaluationExample] = Field(..., min_length=1)


class EvaluationResponse(BaseModel):
    metric_scores: dict[str, float | None]
    rows: list[dict[str, Any]]
    used_ragas: bool
    note: str | None = None
