import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final
from uuid import uuid4

import aiofiles
from fastapi import UploadFile, status

from app.api.schemas.uploads import UploadedDocumentMetadata
from app.config import Settings
from app.services.exceptions import UploadError

logger = logging.getLogger(__name__)

PDF_CONTENT_TYPES: Final[set[str]] = {
    "application/pdf",
    "application/x-pdf",
}
PDF_MAGIC_BYTES: Final[bytes] = b"%PDF-"
READ_CHUNK_SIZE: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class UploadService:
    settings: Settings

    async def save_pdf(self, file: UploadFile) -> UploadedDocumentMetadata:
        original_filename = file.filename or ""
        self._validate_filename(original_filename)
        self._validate_content_type(file.content_type)

        document_id = str(uuid4())
        stored_filename = f"{document_id}.pdf"
        destination = self.settings.upload_dir / stored_filename

        size_bytes = 0
        sha256 = hashlib.sha256()
        header = b""

        try:
            async with aiofiles.open(destination, "wb") as out_file:
                while chunk := await file.read(READ_CHUNK_SIZE):
                    if not header:
                        header = chunk[: len(PDF_MAGIC_BYTES)]

                    size_bytes += len(chunk)
                    if size_bytes > self.settings.max_upload_size_bytes:
                        await out_file.close()
                        destination.unlink(missing_ok=True)
                        raise UploadError(
                            code="file_too_large",
                            message=(
                                "PDF exceeds the configured upload limit of "
                                f"{self.settings.max_upload_size_mb} MB."
                            ),
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        )

                    sha256.update(chunk)
                    await out_file.write(chunk)
        except UploadError:
            raise
        except OSError as exc:
            logger.exception("Failed to save uploaded PDF: %s", exc)
            raise UploadError(
                code="storage_error",
                message="Could not save the uploaded PDF.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc
        finally:
            await file.close()

        if size_bytes == 0:
            destination.unlink(missing_ok=True)
            raise UploadError(code="empty_file", message="Uploaded PDF is empty.")

        if header != PDF_MAGIC_BYTES:
            destination.unlink(missing_ok=True)
            raise UploadError(
                code="invalid_pdf",
                message="Uploaded file does not appear to be a valid PDF.",
            )

        uploaded_at = datetime.now(timezone.utc)
        return UploadedDocumentMetadata(
            document_id=document_id,
            original_filename=Path(original_filename).name,
            stored_filename=stored_filename,
            content_type=file.content_type or "application/pdf",
            size_bytes=size_bytes,
            sha256=sha256.hexdigest(),
            uploaded_at=uploaded_at,
            relative_path=str(destination),
        )

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if not filename:
            raise UploadError(code="missing_filename", message="Uploaded file needs a name.")

        if Path(filename).suffix.lower() != ".pdf":
            raise UploadError(
                code="unsupported_file_type",
                message="Only PDF uploads are supported.",
            )

    @staticmethod
    def _validate_content_type(content_type: str | None) -> None:
        if content_type not in PDF_CONTENT_TYPES:
            raise UploadError(
                code="unsupported_media_type",
                message="Upload content type must be application/pdf.",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )
