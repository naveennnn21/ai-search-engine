import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

from fastapi import status
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.api.schemas.embeddings import (
    ChunkEmbedding,
    DocumentEmbeddingResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    TextEmbedding,
)
from app.config import Settings
from app.services.document_chunking_service import DocumentChunkingService
from app.services.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingService:
    settings: Settings
    document_chunking_service: DocumentChunkingService

    def __post_init__(self) -> None:
        if self.settings.openai_api_key is None:
            return

        api_key = self.settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            return

        object.__setattr__(
            self,
            "_client",
            AsyncOpenAI(api_key=api_key),
        )

    async def embed_texts(self, inputs: list[str]) -> EmbeddingResponse:
        clean_inputs = self._normalize_inputs(inputs)
        embeddings, usage = await self._embed_batches(clean_inputs)

        return EmbeddingResponse(
            model=self.settings.embedding_model,
            input_count=len(clean_inputs),
            embeddings=[
                TextEmbedding(
                    index=index,
                    text=text,
                    embedding=embedding,
                    dimensions=len(embedding),
                )
                for index, (text, embedding) in enumerate(zip(clean_inputs, embeddings))
            ],
            usage=usage,
        )

    async def embed_document_chunks(self, document_id: str) -> DocumentEmbeddingResponse:
        chunked_document = await self.document_chunking_service.chunk_document(document_id)
        chunk_texts = [chunk.text for chunk in chunked_document.chunks]

        if not chunk_texts:
            return DocumentEmbeddingResponse(
                document_id=chunked_document.document_id,
                filename=chunked_document.filename,
                model=self.settings.embedding_model,
                chunk_count=0,
                embeddings=[],
                usage=EmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )

        embeddings, usage = await self._embed_batches(chunk_texts)

        return DocumentEmbeddingResponse(
            document_id=chunked_document.document_id,
            filename=chunked_document.filename,
            model=self.settings.embedding_model,
            chunk_count=len(chunked_document.chunks),
            embeddings=[
                ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=embedding,
                    dimensions=len(embedding),
                    page_numbers=chunk.page_numbers,
                    page_references=chunk.page_references,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": self.settings.embedding_model,
                    },
                )
                for chunk, embedding in zip(chunked_document.chunks, embeddings)
            ],
            usage=usage,
        )

    async def _embed_batches(
        self,
        inputs: list[str],
    ) -> tuple[list[list[float]], EmbeddingUsage]:
        self._ensure_client()

        all_embeddings: list[list[float]] = []
        total_prompt_tokens = 0
        total_tokens = 0

        for batch_start in range(0, len(inputs), self.settings.embedding_batch_size):
            batch = inputs[batch_start : batch_start + self.settings.embedding_batch_size]
            response = await self._create_embeddings_with_retry(batch)
            ordered_data = sorted(response.data, key=lambda item: item.index)

            if len(ordered_data) != len(batch):
                raise EmbeddingError(
                    code="embedding_count_mismatch",
                    message="OpenAI returned a different number of embeddings than requested.",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                )

            all_embeddings.extend([list(item.embedding) for item in ordered_data])
            total_prompt_tokens += response.usage.prompt_tokens
            total_tokens += response.usage.total_tokens

        return all_embeddings, EmbeddingUsage(
            prompt_tokens=total_prompt_tokens,
            total_tokens=total_tokens,
        )

    async def _create_embeddings_with_retry(self, batch: list[str]) -> Any:
        last_error: Exception | None = None

        for attempt in range(1, self.settings.embedding_max_retries + 1):
            try:
                return await self._client.embeddings.create(**self._request_kwargs(batch))
            except Exception as exc:
                if not self._is_retryable_error(exc) or attempt == self.settings.embedding_max_retries:
                    self._raise_embedding_error(exc)

                last_error = exc
                delay_seconds = self._retry_delay(attempt)
                logger.warning(
                    "OpenAI embedding request failed; retrying",
                    extra={
                        "attempt": attempt,
                        "max_retries": self.settings.embedding_max_retries,
                        "delay_seconds": round(delay_seconds, 3),
                        "error_type": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay_seconds)

        self._raise_embedding_error(last_error)

    def _request_kwargs(self, batch: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.embedding_model,
            "input": batch,
            "encoding_format": "float",
        }

        if self.settings.embedding_dimensions is not None:
            kwargs["dimensions"] = self.settings.embedding_dimensions

        return kwargs

    def _ensure_client(self) -> None:
        if not hasattr(self, "_client"):
            raise EmbeddingError(
                code="missing_openai_api_key",
                message="OPENAI_API_KEY is required to generate embeddings.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _retry_delay(self, attempt: int) -> float:
        base_delay = self.settings.embedding_retry_base_seconds * (2 ** (attempt - 1))
        jitter = random.uniform(0, self.settings.embedding_retry_jitter_seconds)
        return min(base_delay + jitter, self.settings.embedding_retry_max_seconds)

    @staticmethod
    def _normalize_inputs(inputs: list[str]) -> list[str]:
        clean_inputs = [text.strip() for text in inputs if text and text.strip()]

        if not clean_inputs:
            raise EmbeddingError(
                code="empty_embedding_input",
                message="At least one non-empty text input is required.",
            )

        return clean_inputs

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True

        if isinstance(exc, APIStatusError):
            return exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS or exc.status_code >= 500

        return False

    @staticmethod
    def _raise_embedding_error(exc: Exception | None) -> None:
        if isinstance(exc, RateLimitError):
            raise EmbeddingError(
                code="openai_rate_limit",
                message="OpenAI embeddings API rate limit was exceeded.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            ) from exc

        if isinstance(exc, APIStatusError):
            status_code = (
                status.HTTP_502_BAD_GATEWAY
                if exc.status_code >= 500
                else status.HTTP_400_BAD_REQUEST
            )
            raise EmbeddingError(
                code="openai_api_error",
                message=f"OpenAI embeddings API returned status {exc.status_code}.",
                status_code=status_code,
            ) from exc

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            raise EmbeddingError(
                code="openai_connection_error",
                message="Could not reach the OpenAI embeddings API.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc

        raise EmbeddingError(
            code="embedding_generation_error",
            message="Could not generate embeddings.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
