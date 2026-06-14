import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.api.schemas.embeddings import EmbeddingUsage
from app.api.schemas.retrieval import RetrievalMatch, RetrievalResponse
from app.api.schemas.vector_store import VectorMetadata
from app.config import Settings
from app.services.llm_service import LlmService
from app.services.vector_store_service import VectorStoreService

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


@dataclass
class CandidateScore:
    metadata: VectorMetadata
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float | None = None

    @property
    def hybrid_score(self) -> float:
        return self.vector_score + self.bm25_score


@dataclass(frozen=True)
class RetrievalService:
    settings: Settings
    vector_store_service: VectorStoreService
    llm_service: LlmService

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_query_expansion: bool = True,
        use_reranking: bool = True,
    ) -> RetrievalResponse:
        normalized_query = query.strip()
        expanded_queries = (
            await self.expand_query(normalized_query) if use_query_expansion else [normalized_query]
        )

        vector_candidates, usage = await self._vector_candidates(expanded_queries)
        bm25_scores = self._bm25_scores(expanded_queries)

        candidates: dict[str, CandidateScore] = {}
        for chunk_id, score in vector_candidates.items():
            metadata = self.vector_store_service.get_metadata_by_chunk_id(chunk_id)
            if metadata is not None:
                candidates[chunk_id] = CandidateScore(
                    metadata=metadata,
                    vector_score=score * self.settings.vector_search_weight,
                )

        for chunk_id, score in bm25_scores.items():
            metadata = self.vector_store_service.get_metadata_by_chunk_id(chunk_id)
            if metadata is None:
                continue
            candidate = candidates.setdefault(chunk_id, CandidateScore(metadata=metadata))
            candidate.bm25_score = score * self.settings.bm25_search_weight

        ranked = sorted(candidates.values(), key=lambda item: item.hybrid_score, reverse=True)
        rerank_pool = ranked[: max(top_k, self.settings.rerank_candidate_count)]

        if use_reranking and rerank_pool:
            ranked = await self._rerank(normalized_query, rerank_pool)
        else:
            ranked = rerank_pool

        matches = [
            self._to_match(rank, candidate)
            for rank, candidate in enumerate(ranked[:top_k], start=1)
        ]

        return RetrievalResponse(
            query=normalized_query,
            expanded_queries=expanded_queries,
            top_k=top_k,
            total_candidates=len(candidates),
            matches=matches,
            usage=usage,
        )

    async def expand_query(self, query: str) -> list[str]:
        if not self.settings.enable_query_expansion:
            return [query]

        prompt = (
            "Generate concise semantic search query expansions for the user query.\n"
            "Return JSON only as an array of strings. Include the original query first. "
            f"Return at most {self.settings.query_expansion_count + 1} queries.\n\n"
            f"Query: {query}"
        )

        try:
            data = await self.llm_service.generate_json(prompt)
            expansions = [str(item).strip() for item in data if str(item).strip()]
        except Exception:
            expansions = []

        deduped: list[str] = []
        for item in [query, *expansions]:
            if item.lower() not in {existing.lower() for existing in deduped}:
                deduped.append(item)

        return deduped[: self.settings.query_expansion_count + 1]

    async def _vector_candidates(self, queries: list[str]) -> tuple[dict[str, float], EmbeddingUsage]:
        scores: dict[str, float] = {}
        total_usage = EmbeddingUsage(prompt_tokens=0, total_tokens=0)

        for expanded_query in queries:
            result = await self.vector_store_service.search_text(
                expanded_query,
                self.settings.retrieval_candidate_count,
            )
            total_usage.prompt_tokens += result.usage.prompt_tokens
            total_usage.total_tokens += result.usage.total_tokens

            for match in result.matches:
                normalized_score = max(0.0, min(1.0, (match.score + 1.0) / 2.0))
                scores[match.chunk_id] = max(scores.get(match.chunk_id, 0.0), normalized_score)

        return scores, total_usage

    def _bm25_scores(self, queries: list[str]) -> dict[str, float]:
        documents = self.vector_store_service.all_metadata()
        if not documents:
            return {}

        tokenized_docs = [self._tokens(item.text) for item in documents]
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths)

        document_frequency: dict[str, int] = defaultdict(int)
        for tokens in tokenized_docs:
            for token in set(tokens):
                document_frequency[token] += 1

        raw_scores: dict[str, float] = defaultdict(float)
        query_tokens = self._tokens(" ".join(queries))
        query_terms = Counter(query_tokens)

        k1 = 1.5
        b = 0.75
        total_documents = len(documents)

        for metadata, tokens, doc_length in zip(documents, tokenized_docs, doc_lengths):
            term_counts = Counter(tokens)
            score = 0.0

            for term, query_weight in query_terms.items():
                if term not in term_counts:
                    continue

                df = document_frequency.get(term, 0)
                idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                tf = term_counts[term]
                denominator = tf + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1))
                score += query_weight * idf * ((tf * (k1 + 1)) / denominator)

            raw_scores[metadata.chunk_id] = score

        return self._normalize_scores(raw_scores)

    async def _rerank(
        self,
        query: str,
        candidates: list[CandidateScore],
    ) -> list[CandidateScore]:
        if not self.settings.enable_reranking:
            return sorted(candidates, key=lambda item: item.hybrid_score, reverse=True)

        candidate_lines = "\n\n".join(
            f"{index}. chunk_id={candidate.metadata.chunk_id}\n{candidate.metadata.text[:1200]}"
            for index, candidate in enumerate(candidates, start=1)
        )
        prompt = (
            "Rerank these retrieved chunks for answering the query. "
            "Return JSON only as an array of objects with chunk_id and score from 0 to 1.\n\n"
            f"Query: {query}\n\nChunks:\n{candidate_lines}"
        )

        try:
            data = await self.llm_service.generate_json(prompt)
            score_by_chunk_id = {
                str(item["chunk_id"]): float(item["score"])
                for item in data
                if "chunk_id" in item and "score" in item
            }
            for candidate in candidates:
                if candidate.metadata.chunk_id in score_by_chunk_id:
                    candidate.rerank_score = max(0.0, min(1.0, score_by_chunk_id[candidate.metadata.chunk_id]))
        except Exception:
            for candidate in candidates:
                candidate.rerank_score = candidate.hybrid_score

        return sorted(
            candidates,
            key=lambda item: item.rerank_score if item.rerank_score is not None else item.hybrid_score,
            reverse=True,
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}

        maximum = max(scores.values())
        if maximum <= 0:
            return {key: 0.0 for key in scores}

        return {key: value / maximum for key, value in scores.items()}

    @staticmethod
    def _to_match(rank: int, candidate: CandidateScore) -> RetrievalMatch:
        final_score = (
            candidate.rerank_score
            if candidate.rerank_score is not None
            else candidate.hybrid_score
        )
        metadata = candidate.metadata
        return RetrievalMatch(
            rank=rank,
            chunk_id=metadata.chunk_id,
            document_id=metadata.document_id,
            chunk_index=metadata.chunk_index,
            text=metadata.text,
            similarity_score=max(0.0, min(1.0, final_score)),
            vector_score=candidate.vector_score,
            bm25_score=candidate.bm25_score,
            rerank_score=candidate.rerank_score,
            page_numbers=metadata.page_numbers,
            page_references=metadata.page_references,
            metadata=metadata.metadata,
        )
