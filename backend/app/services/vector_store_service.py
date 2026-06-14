import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np
from anyio import to_thread
from fastapi import status

from app.api.schemas.embeddings import EmbeddingUsage
from app.api.schemas.vector_store import (
    VectorMetadata,
    VectorSearchMatch,
    VectorSearchResponse,
    VectorStoreLoadResponse,
    VectorStoreSaveResponse,
    VectorStoreStatsResponse,
    VectorUpsertResponse,
)
from app.config import Settings
from app.services.embedding_service import EmbeddingService
from app.services.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreState:
    index: faiss.IndexIDMap2 | None = None
    dimension: int | None = None
    next_vector_id: int = 0
    metadata_by_id: dict[int, VectorMetadata] = field(default_factory=dict)
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class VectorStoreService:
    settings: Settings
    embedding_service: EmbeddingService

    def __post_init__(self) -> None:
        self.settings.vector_index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._state = VectorStoreState()

    async def index_document(self, document_id: str) -> VectorUpsertResponse:
        document_embeddings = await self.embedding_service.embed_document_chunks(document_id)
        inserted_count, skipped_count = await to_thread.run_sync(
            self._insert_document_embeddings_sync,
            document_embeddings,
        )
        await self.save()

        return VectorUpsertResponse(
            document_id=document_embeddings.document_id,
            filename=document_embeddings.filename,
            inserted_count=inserted_count,
            skipped_count=skipped_count,
            total_vectors=self._total_vectors(),
            index_path=str(self._index_path),
            metadata_path=str(self._metadata_path),
            usage=document_embeddings.usage,
        )

    async def search_text(self, query: str, top_k: int) -> VectorSearchResponse:
        embedding_response = await self.embedding_service.embed_texts([query])
        query_embedding = embedding_response.embeddings[0].embedding
        matches = await to_thread.run_sync(self._search_sync, query_embedding, top_k)

        return VectorSearchResponse(
            query=query,
            model=embedding_response.model,
            top_k=top_k,
            total_vectors=self._total_vectors(),
            matches=matches,
            usage=embedding_response.usage,
        )

    async def save(self) -> VectorStoreSaveResponse:
        await to_thread.run_sync(self._save_sync)
        return VectorStoreSaveResponse(
            saved=True,
            total_vectors=self._total_vectors(),
            index_path=str(self._index_path),
            metadata_path=str(self._metadata_path),
        )

    async def load(self) -> VectorStoreLoadResponse:
        await to_thread.run_sync(self._load_sync)
        return VectorStoreLoadResponse(
            loaded=True,
            total_vectors=self._total_vectors(),
            index_path=str(self._index_path),
            metadata_path=str(self._metadata_path),
        )

    def stats(self) -> VectorStoreStatsResponse:
        with self._lock:
            return VectorStoreStatsResponse(
                dimension=self._state.dimension,
                total_vectors=len(self._state.metadata_by_id),
                index_path=str(self._index_path),
                metadata_path=str(self._metadata_path),
            )

    def all_metadata(self) -> list[VectorMetadata]:
        with self._lock:
            return list(self._state.metadata_by_id.values())

    def get_metadata_by_chunk_id(self, chunk_id: str) -> VectorMetadata | None:
        with self._lock:
            for metadata in self._state.metadata_by_id.values():
                if metadata.chunk_id == chunk_id:
                    return metadata
            return None

    def _insert_document_embeddings_sync(self, document_embeddings: Any) -> tuple[int, int]:
        inserted_vectors: list[list[float]] = []
        inserted_ids: list[int] = []
        inserted_metadata: list[VectorMetadata] = []
        skipped_count = 0

        with self._lock:
            for chunk_embedding in document_embeddings.embeddings:
                if chunk_embedding.chunk_id in self._state.chunk_ids:
                    skipped_count += 1
                    continue

                vector = chunk_embedding.embedding
                self._ensure_index(len(vector))

                vector_id = self._state.next_vector_id
                self._state.next_vector_id += 1

                metadata = VectorMetadata(
                    vector_id=vector_id,
                    chunk_id=chunk_embedding.chunk_id,
                    document_id=chunk_embedding.document_id,
                    chunk_index=chunk_embedding.chunk_index,
                    text=chunk_embedding.text,
                    page_numbers=chunk_embedding.page_numbers,
                    page_references=chunk_embedding.page_references,
                    metadata=chunk_embedding.metadata,
                )

                inserted_vectors.append(vector)
                inserted_ids.append(vector_id)
                inserted_metadata.append(metadata)

            if inserted_vectors:
                vectors_array = self._normalize_vectors(inserted_vectors)
                ids_array = np.asarray(inserted_ids, dtype=np.int64)
                self._state.index.add_with_ids(vectors_array, ids_array)

                for metadata in inserted_metadata:
                    self._state.metadata_by_id[metadata.vector_id] = metadata
                    self._state.chunk_ids.add(metadata.chunk_id)

        return len(inserted_vectors), skipped_count

    def _search_sync(self, query_embedding: list[float], top_k: int) -> list[VectorSearchMatch]:
        with self._lock:
            if self._state.index is None or not self._state.metadata_by_id:
                raise VectorStoreError(
                    code="empty_vector_index",
                    message="The FAISS index is empty. Insert document vectors before searching.",
                    status_code=status.HTTP_404_NOT_FOUND,
                )

            if len(query_embedding) != self._state.dimension:
                raise VectorStoreError(
                    code="dimension_mismatch",
                    message="Query embedding dimension does not match the FAISS index.",
                    status_code=status.HTTP_400_BAD_REQUEST,
                )

            query_array = self._normalize_vectors([query_embedding])
            search_k = min(top_k, len(self._state.metadata_by_id))
            scores, vector_ids = self._state.index.search(query_array, search_k)

            matches: list[VectorSearchMatch] = []
            for rank, (score, vector_id) in enumerate(zip(scores[0], vector_ids[0]), start=1):
                if vector_id < 0:
                    continue

                metadata = self._state.metadata_by_id.get(int(vector_id))
                if metadata is None:
                    logger.warning("FAISS returned vector ID without metadata: %s", vector_id)
                    continue

                matches.append(
                    VectorSearchMatch(
                        rank=rank,
                        score=float(score),
                        vector_id=metadata.vector_id,
                        chunk_id=metadata.chunk_id,
                        document_id=metadata.document_id,
                        chunk_index=metadata.chunk_index,
                        text=metadata.text,
                        page_numbers=metadata.page_numbers,
                        page_references=metadata.page_references,
                        metadata=metadata.metadata,
                    )
                )

            return matches

    def _save_sync(self) -> None:
        with self._lock:
            if self._state.index is not None:
                faiss.write_index(self._state.index, str(self._index_path))

            metadata_payload = {
                "dimension": self._state.dimension,
                "next_vector_id": self._state.next_vector_id,
                "vectors": [
                    metadata.model_dump(mode="json")
                    for metadata in self._state.metadata_by_id.values()
                ],
            }

            self._metadata_path.write_text(
                json.dumps(metadata_payload, indent=2),
                encoding="utf-8",
            )

    def _load_sync(self) -> None:
        with self._lock:
            if not self._index_path.exists() or not self._metadata_path.exists():
                raise VectorStoreError(
                    code="vector_store_not_found",
                    message="No saved FAISS index and metadata were found.",
                    status_code=status.HTTP_404_NOT_FOUND,
                )

            try:
                index = faiss.read_index(str(self._index_path))
                metadata_payload = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.exception("Failed to load FAISS index or metadata: %s", exc)
                raise VectorStoreError(
                    code="vector_store_load_error",
                    message="Could not load the saved FAISS index.",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                ) from exc

            metadata_by_id = {
                item["vector_id"]: VectorMetadata.model_validate(item)
                for item in metadata_payload.get("vectors", [])
            }
            dimension = metadata_payload.get("dimension")

            if index.ntotal != len(metadata_by_id):
                raise VectorStoreError(
                    code="vector_store_corrupt",
                    message="Saved FAISS index and metadata counts do not match.",
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            self._state = VectorStoreState(
                index=index,
                dimension=dimension,
                next_vector_id=metadata_payload.get("next_vector_id", len(metadata_by_id)),
                metadata_by_id=metadata_by_id,
                chunk_ids={metadata.chunk_id for metadata in metadata_by_id.values()},
            )

    def _ensure_index(self, dimension: int) -> None:
        if self._state.index is None:
            base_index = faiss.IndexFlatIP(dimension)
            self._state.index = faiss.IndexIDMap2(base_index)
            self._state.dimension = dimension
            return

        if self._state.dimension != dimension:
            raise VectorStoreError(
                code="dimension_mismatch",
                message="Inserted vector dimension does not match the existing FAISS index.",
                status_code=status.HTTP_400_BAD_REQUEST,
            )

    @staticmethod
    def _normalize_vectors(vectors: list[list[float]]) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)

        if array.ndim != 2:
            raise VectorStoreError(
                code="invalid_vector_shape",
                message="Vectors must be a two-dimensional array.",
            )

        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise VectorStoreError(
                code="zero_vector",
                message="Cannot insert or search with a zero-length vector.",
            )

        return array / norms

    def _total_vectors(self) -> int:
        with self._lock:
            return len(self._state.metadata_by_id)

    @property
    def _index_path(self) -> Path:
        return self.settings.vector_index_dir / self.settings.faiss_index_filename

    @property
    def _metadata_path(self) -> Path:
        return self.settings.vector_index_dir / self.settings.vector_metadata_filename
