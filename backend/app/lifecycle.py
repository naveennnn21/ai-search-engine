from fastapi import FastAPI

from app.config import get_settings
from app.services.document_chunking_service import DocumentChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import LlmService
from app.services.pdf_processing_service import PdfProcessingService
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService
from app.services.upload_service import UploadService
from app.services.vector_store_service import VectorStoreService


def configure_services(app: FastAPI) -> None:
    settings = get_settings()
    app.state.upload_service = UploadService(settings=settings)
    app.state.pdf_processing_service = PdfProcessingService(settings=settings)
    app.state.document_chunking_service = DocumentChunkingService(
        settings=settings,
        pdf_processing_service=app.state.pdf_processing_service,
    )
    app.state.embedding_service = EmbeddingService(
        settings=settings,
        document_chunking_service=app.state.document_chunking_service,
    )
    app.state.vector_store_service = VectorStoreService(
        settings=settings,
        embedding_service=app.state.embedding_service,
    )
    app.state.llm_service = LlmService(settings=settings)
    app.state.retrieval_service = RetrievalService(
        settings=settings,
        vector_store_service=app.state.vector_store_service,
        llm_service=app.state.llm_service,
    )
    app.state.rag_service = RagService(
        retrieval_service=app.state.retrieval_service,
        llm_service=app.state.llm_service,
    )
    app.state.evaluation_service = EvaluationService()
