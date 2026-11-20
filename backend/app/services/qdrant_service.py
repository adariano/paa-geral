import time
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from backend.app.config import settings
from backend.app.schemas import Citation


class SemanticSearchService:
    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._encoder: SentenceTransformer | None = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        return self._client

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(settings.embedding_model)
        return self._encoder

    def search(self, query: str, top_k: int) -> tuple[list[Citation], float, float]:
        started = time.perf_counter()
        vector = self.encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()
        embedding_ms = (time.perf_counter() - started) * 1000

        search_started = time.perf_counter()
        response = self.client.query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        search_ms = (time.perf_counter() - search_started) * 1000

        citations: list[Citation] = []
        for hit in response.points:
            payload: dict[str, Any] = hit.payload or {}
            citations.append(
                Citation(
                    document=str(payload.get("title") or "Documento sem título"),
                    source=str(payload.get("source") or "Base Local"),
                    source_url=(str(payload.get("source_url")).strip() or None)
                    if payload.get("source_url")
                    else None,
                    doc_id=str(payload.get("doc_id")) if payload.get("doc_id") is not None else None,
                    snippet=str(payload.get("text") or ""),
                    score=round(float(hit.score), 4),
                )
            )
        return citations, round(embedding_ms, 2), round(search_ms, 2)


search_service = SemanticSearchService()
