from dataclasses import dataclass

from anyio import to_thread

from app.api.schemas.evaluation import EvaluationExample, EvaluationResponse


@dataclass(frozen=True)
class EvaluationService:
    async def evaluate(self, examples: list[EvaluationExample]) -> EvaluationResponse:
        return await to_thread.run_sync(self._evaluate_sync, examples)

    @staticmethod
    def _evaluate_sync(examples: list[EvaluationExample]) -> EvaluationResponse:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
        except ImportError:
            return EvaluationResponse(
                metric_scores={
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                },
                rows=[],
                used_ragas=False,
                note="Install ragas and datasets to run RAGAS evaluation.",
            )

        rows = [
            {
                "question": example.question,
                "answer": example.answer,
                "contexts": example.contexts,
                "ground_truth": example.ground_truth or "",
            }
            for example in examples
        ]
        dataset = Dataset.from_list(rows)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        dataframe = result.to_pandas()
        metric_scores = {
            column: float(dataframe[column].mean())
            for column in dataframe.columns
            if column not in {"question", "answer", "contexts", "ground_truth"}
        }

        return EvaluationResponse(
            metric_scores=metric_scores,
            rows=dataframe.to_dict(orient="records"),
            used_ragas=True,
            note=None,
        )
