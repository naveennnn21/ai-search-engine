import logging
from dataclasses import dataclass

from fastapi import status

from app.api.schemas.documents import (
    ChunkedDocumentResponse,
    DocumentChunk,
    ExtractedDocumentResponse,
    PageReference,
)
from app.config import Settings
from app.services.exceptions import DocumentChunkingError
from app.services.pdf_processing_service import PdfProcessingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageSpan:
    page_number: int
    global_start: int
    global_end: int


@dataclass(frozen=True)
class DocumentChunkingService:
    settings: Settings
    pdf_processing_service: PdfProcessingService

    async def chunk_document(self, document_id: str) -> ChunkedDocumentResponse:
        if self.settings.chunk_overlap >= self.settings.chunk_size:
            raise DocumentChunkingError(
                code="invalid_chunk_settings",
                message="Chunk overlap must be smaller than chunk size.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        extracted_document = await self.pdf_processing_service.extract_text(document_id)

        try:
            return self._chunk_extracted_document(extracted_document)
        except DocumentChunkingError:
            raise
        except Exception as exc:
            logger.exception("Unexpected chunking error for %s: %s", document_id, exc)
            raise DocumentChunkingError(
                code="chunking_error",
                message="Could not chunk the extracted document text.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def _chunk_extracted_document(
        self,
        document: ExtractedDocumentResponse,
    ) -> ChunkedDocumentResponse:
        full_text, page_spans = self._build_page_aware_text(document)

        if not full_text:
            return ChunkedDocumentResponse(
                document_id=document.document_id,
                filename=document.filename,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                chunk_count=0,
                chunks=[],
            )

        chunks: list[DocumentChunk] = []
        start = 0
        chunk_index = 0

        while start < len(full_text):
            end = self._find_chunk_end(full_text, start)
            chunk_text = full_text[start:end].strip()

            if chunk_text:
                references = self._page_references_for_span(page_spans, start, end)
                page_numbers = [reference.page_number for reference in references]
                chunk_id = f"{document.document_id}:{chunk_index}"

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        text=chunk_text,
                        character_count=len(chunk_text),
                        start_character=start,
                        end_character=end,
                        page_numbers=page_numbers,
                        page_references=references,
                        metadata={
                            "filename": document.filename,
                            "chunk_size": self.settings.chunk_size,
                            "chunk_overlap": self.settings.chunk_overlap,
                            "source": "pdf",
                        },
                    )
                )
                chunk_index += 1

            if end >= len(full_text):
                break

            next_start = max(end - self.settings.chunk_overlap, start + 1)
            start = self._advance_to_text(full_text, next_start)

        return ChunkedDocumentResponse(
            document_id=document.document_id,
            filename=document.filename,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            chunk_count=len(chunks),
            chunks=chunks,
        )

    @staticmethod
    def _build_page_aware_text(
        document: ExtractedDocumentResponse,
    ) -> tuple[str, list[PageSpan]]:
        text_parts: list[str] = []
        page_spans: list[PageSpan] = []
        cursor = 0

        for page in document.pages:
            if text_parts:
                separator = "\n\n"
                text_parts.append(separator)
                cursor += len(separator)

            page_text = page.text.strip()
            page_start = cursor
            text_parts.append(page_text)
            cursor += len(page_text)
            page_spans.append(
                PageSpan(
                    page_number=page.page_number,
                    global_start=page_start,
                    global_end=cursor,
                )
            )

        return "".join(text_parts), page_spans

    def _find_chunk_end(self, text: str, start: int) -> int:
        target_end = min(start + self.settings.chunk_size, len(text))

        if target_end == len(text):
            return target_end

        minimum_end = start + max(self.settings.chunk_size // 2, 1)
        search_window = text[minimum_end:target_end]

        for separator in ("\n\n", "\n", ". ", " "):
            separator_index = search_window.rfind(separator)
            if separator_index != -1:
                return minimum_end + separator_index + len(separator)

        return target_end

    @staticmethod
    def _advance_to_text(text: str, start: int) -> int:
        while start < len(text) and text[start].isspace():
            start += 1
        return start

    @staticmethod
    def _page_references_for_span(
        page_spans: list[PageSpan],
        chunk_start: int,
        chunk_end: int,
    ) -> list[PageReference]:
        references: list[PageReference] = []

        for page_span in page_spans:
            overlap_start = max(chunk_start, page_span.global_start)
            overlap_end = min(chunk_end, page_span.global_end)

            if overlap_start < overlap_end:
                references.append(
                    PageReference(
                        page_number=page_span.page_number,
                        start_character=overlap_start - page_span.global_start,
                        end_character=overlap_end - page_span.global_start,
                    )
                )

        return references
