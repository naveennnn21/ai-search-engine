import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.exceptions import (
    DocumentChunkingError,
    DocumentProcessingError,
    EmbeddingError,
    LlmError,
    RagError,
    RetrievalError,
    UploadError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(UploadError)
    async def upload_error_handler(_: Request, exc: UploadError) -> JSONResponse:
        logger.warning("Upload rejected: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(DocumentProcessingError)
    async def document_processing_error_handler(
        _: Request,
        exc: DocumentProcessingError,
    ) -> JSONResponse:
        logger.warning("Document processing failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(DocumentChunkingError)
    async def document_chunking_error_handler(
        _: Request,
        exc: DocumentChunkingError,
    ) -> JSONResponse:
        logger.warning("Document chunking failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(EmbeddingError)
    async def embedding_error_handler(_: Request, exc: EmbeddingError) -> JSONResponse:
        logger.warning("Embedding generation failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(VectorStoreError)
    async def vector_store_error_handler(_: Request, exc: VectorStoreError) -> JSONResponse:
        logger.warning("Vector store operation failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(_: Request, exc: RetrievalError) -> JSONResponse:
        logger.warning("Retrieval failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(LlmError)
    async def llm_error_handler(_: Request, exc: LlmError) -> JSONResponse:
        logger.warning("LLM operation failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RagError)
    async def rag_error_handler(_: Request, exc: RagError) -> JSONResponse:
        logger.warning("RAG operation failed: %s", exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": exc.errors(),
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        logger.warning("HTTP error: %s", exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": exc.detail}},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_server_error",
                    "message": "An unexpected error occurred.",
                }
            },
        )
