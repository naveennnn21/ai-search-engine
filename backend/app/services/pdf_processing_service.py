import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from anyio import to_thread
from fastapi import status
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.api.schemas.documents import ExtractedDocumentResponse, ExtractedPage
from app.config import Settings
from app.services.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

DOCUMENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdfProcessingService:
    settings: Settings

    async def extract_text(self, document_id: str) -> ExtractedDocumentResponse:
        self._validate_document_id(document_id)
        pdf_path = self.settings.upload_dir / f"{document_id}.pdf"

        if not pdf_path.exists():
            raise DocumentProcessingError(
                code="document_not_found",
                message="Uploaded PDF was not found.",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        if not pdf_path.is_file():
            raise DocumentProcessingError(
                code="invalid_document_path",
                message="Document path does not point to a PDF file.",
            )

        try:
            return await to_thread.run_sync(self._extract_text_sync, document_id, pdf_path)
        except DocumentProcessingError:
            raise
        except PdfReadError as exc:
            logger.warning("Failed to read PDF %s: %s", document_id, exc)
            raise DocumentProcessingError(
                code="pdf_read_error",
                message="Could not read the uploaded PDF.",
            ) from exc
        except OSError as exc:
            logger.exception("Failed to open PDF %s: %s", document_id, exc)
            raise DocumentProcessingError(
                code="document_io_error",
                message="Could not open the uploaded PDF.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected PDF processing error for %s: %s", document_id, exc)
            raise DocumentProcessingError(
                code="pdf_processing_error",
                message="Could not process the uploaded PDF.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def _extract_text_sync(
        self,
        document_id: str,
        pdf_path: Path,
    ) -> ExtractedDocumentResponse:
        reader = PdfReader(str(pdf_path))

        if reader.is_encrypted:
            raise DocumentProcessingError(
                code="encrypted_pdf",
                message="Encrypted PDFs are not supported yet.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        pages: list[ExtractedPage] = []
        total_characters = 0

        for page_index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "Text extraction failed for document %s page %s: %s",
                    document_id,
                    page_index,
                    exc,
                )
                raise DocumentProcessingError(
                    code="page_extraction_error",
                    message=f"Could not extract text from page {page_index}.",
                ) from exc

            normalized_text = self._normalize_text(text)
            character_count = len(normalized_text)
            total_characters += character_count
            pages.append(
                ExtractedPage(
                    page_number=page_index,
                    text=normalized_text,
                    character_count=character_count,
                )
            )

        return ExtractedDocumentResponse(
            document_id=document_id,
            filename=pdf_path.name,
            page_count=len(pages),
            total_characters=total_characters,
            pages=pages,
        )

    @staticmethod
    def _validate_document_id(document_id: str) -> None:
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise DocumentProcessingError(
                code="invalid_document_id",
                message="Document ID must be a valid UUID.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [line.strip() for line in text.replace("\x00", "").splitlines()]
        non_empty_lines = [line for line in lines if line]
        return "\n".join(non_empty_lines)
