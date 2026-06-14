import asyncio
import json
import logging
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

from app.config import Settings
from app.services.exceptions import LlmError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LlmService:
    settings: Settings

    def __post_init__(self) -> None:
        if self.settings.openai_api_key is None:
            return

        api_key = self.settings.openai_api_key.get_secret_value().strip()
        if not api_key:
            return

        object.__setattr__(self, "_client", AsyncOpenAI(api_key=api_key))

    async def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        self._ensure_client()
        response = await self._create_response_with_retry(prompt, temperature)
        return self._extract_text(response)

    async def generate_json(self, prompt: str, temperature: float = 0.1) -> Any:
        text = await self.generate_text(prompt, temperature)
        try:
            return json.loads(self._strip_code_fence(text))
        except json.JSONDecodeError as exc:
            raise LlmError(
                code="invalid_llm_json",
                message="The LLM did not return valid JSON.",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

    async def _create_response_with_retry(self, prompt: str, temperature: float) -> Any:
        last_error: Exception | None = None

        for attempt in range(1, self.settings.llm_max_retries + 1):
            try:
                return await self._client.responses.create(
                    model=self.settings.llm_model,
                    input=prompt,
                    temperature=temperature,
                )
            except Exception as exc:
                if not self._is_retryable_error(exc) or attempt == self.settings.llm_max_retries:
                    self._raise_llm_error(exc)

                last_error = exc
                delay = min(
                    self.settings.llm_retry_base_seconds * (2 ** (attempt - 1)),
                    self.settings.llm_retry_max_seconds,
                )
                logger.warning(
                    "OpenAI LLM request failed; retrying",
                    extra={"attempt": attempt, "delay_seconds": delay, "error": type(exc).__name__},
                )
                await asyncio.sleep(delay)

        self._raise_llm_error(last_error)

    def _ensure_client(self) -> None:
        if not hasattr(self, "_client"):
            raise LlmError(
                code="missing_openai_api_key",
                message="OPENAI_API_KEY is required for LLM generation.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @staticmethod
    def _extract_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text).strip()

        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1]
            stripped = stripped.rsplit("```", 1)[0]
        return stripped.strip()

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS or exc.status_code >= 500
        return False

    @staticmethod
    def _raise_llm_error(exc: Exception | None) -> None:
        if isinstance(exc, RateLimitError):
            raise LlmError(
                code="openai_rate_limit",
                message="OpenAI rate limit was exceeded.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            ) from exc

        if isinstance(exc, APIStatusError):
            raise LlmError(
                code="openai_api_error",
                message=f"OpenAI API returned status {exc.status_code}.",
                status_code=status.HTTP_502_BAD_GATEWAY,
            ) from exc

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            raise LlmError(
                code="openai_connection_error",
                message="Could not reach OpenAI.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ) from exc

        raise LlmError(
            code="llm_generation_error",
            message="Could not generate an LLM response.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from exc
