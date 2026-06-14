from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.embeddings import router as embeddings_router
from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router
from app.api.routes.retrieval import router as retrieval_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.vector_store import router as vector_store_router
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.lifecycle import configure_services


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="AI Search Engine API",
        description="Production-grade AI search backend.",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    configure_services(app)
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(embeddings_router, prefix="/api/v1")
    app.include_router(evaluation_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")
    app.include_router(retrieval_router, prefix="/api/v1")
    app.include_router(vector_store_router, prefix="/api/v1")
    app.include_router(uploads_router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
