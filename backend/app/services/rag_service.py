from dataclasses import dataclass

from app.api.schemas.rag import Citation, RagResponse
from app.services.llm_service import LlmService
from app.services.retrieval_service import RetrievalService


@dataclass(frozen=True)
class RagService:
    retrieval_service: RetrievalService
    llm_service: LlmService

    async def answer(self, query: str, top_k: int) -> RagResponse:
        retrieved = await self.retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            use_query_expansion=True,
            use_reranking=True,
        )
        citations = self._citations(retrieved.matches)
        prompt = self._build_prompt(query, citations)
        answer = await self.llm_service.generate_text(prompt, temperature=0.2)

        return RagResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved.matches,
        )

    @staticmethod
    def _citations(matches: list) -> list[Citation]:
        citations: list[Citation] = []
        for index, match in enumerate(matches, start=1):
            citations.append(
                Citation(
                    source_id=f"SOURCE {index}",
                    document_id=match.document_id,
                    chunk_id=match.chunk_id,
                    page_numbers=match.page_numbers,
                    snippet=match.text[:500],
                )
            )
        return citations

    @staticmethod
    def _build_prompt(query: str, citations: list[Citation]) -> str:
        context = "\n\n".join(
            (
                f"[{citation.source_id}]\n"
                f"Document: {citation.document_id}\n"
                f"Pages: {', '.join(str(page) for page in citation.page_numbers)}\n"
                f"Text: {citation.snippet}"
            )
            for citation in citations
        )
        return (
            "You are a careful RAG assistant. Answer only from the provided sources. "
            "If the sources do not contain enough information, say that the documents do not provide enough evidence. "
            "Cite sources inline using [SOURCE N].\n\n"
            f"Question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Answer:"
        )
