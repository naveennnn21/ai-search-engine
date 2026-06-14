import logging

from fastapi import APIRouter, File, Request, UploadFile, status

from app.api.schemas.uploads import UploadResponse
from app.services.upload_service import UploadService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF document",
)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
) -> UploadResponse:
    upload_service: UploadService = request.app.state.upload_service
    metadata = await upload_service.save_pdf(file)
    logger.info(
        "PDF uploaded successfully",
        extra={
            "document_id": metadata.document_id,
            "filename": metadata.original_filename,
            "size_bytes": metadata.size_bytes,
        },
    )
    return UploadResponse(document=metadata)
