from fastapi import FastAPI

from app.config import get_settings
from app.services.upload_service import UploadService


def configure_services(app: FastAPI) -> None:
    settings = get_settings()
    app.state.upload_service = UploadService(settings=settings)
