from fastapi import FastAPI

from app.api.routes.uploads import router as uploads_router
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.lifecycle import configure_services


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="AI Search Engine API",
        description="Production-grade AI search backend.",
        version="0.1.0",
    )

    register_exception_handlers(app)
    configure_services(app)
    app.include_router(uploads_router, prefix="/api/v1")

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
