import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.schemas import AnswerResponse, Citation, QuestionRequest, Timings
from backend.app.services.llm_service import generate_answer
from backend.app.services.qdrant_service import search_service

app = FastAPI(
    title="PAA — Busca Semântica em História Brasileira",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "mock_mode": settings.mock_mode,
        "model": settings.ollama_model,
        "collection": settings.qdrant_collection,
    }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionRequest) -> AnswerResponse:
    total_start = time.perf_counter()
    top_k = payload.top_k or settings.top_k_results

    if settings.mock_mode:
        mock_citations = [
            Citation(
                document="Proclamação da República do Brasil",
                source="Amostra Mock (Wikipedia PT)",
                source_url="https://pt.wikipedia.org/wiki/Proclama%C3%A7%C3%A3o_da_Rep%C3%BAblica_do_Brasil",
                doc_id="mock-republica",
                snippet="A Proclamação da República do Brasil foi um levante político-militar ocorrido em 15 de novembro de 1889 que instaurou a forma republicana federativa presidencialista.",
                score=0.92,
            ),
            Citation(
                document="Lei Áurea",
                source="Amostra Mock (Wikipedia PT)",
                source_url="https://pt.wikipedia.org/wiki/Lei_%C3%81urea",
                doc_id="mock-lei-aurea",
                snippet="A Lei Áurea, sancionada em 13 de maio de 1888 pela Princesa Isabel, extinguiu oficialmente a escravidão no Brasil.",
                score=0.78,
            ),
        ]
        total_ms = round((time.perf_counter() - total_start) * 1000, 2)
        return AnswerResponse(
            answer=(
                f"[MODO MOCK] Resposta simulada para: '{payload.question}'. "
                "A Proclamação da República ocorreu em 15 de novembro de 1889 [Fonte 1], "
                "pouco após a abolição formal da escravidão pela Lei Áurea em 1888 [Fonte 2]."
            ),
            citations=mock_citations[:top_k],
            timings=Timings(
                embedding_ms=0.0,
                search_ms=0.0,
                llm_ms=0.0,
                total_ms=total_ms,
            ),
            model=f"{settings.ollama_model} (mock)",
        )

    try:
        # Busca e geração de embeddings são síncronas; execute-as fora do event loop.
        citations, embedding_ms, search_ms = await asyncio.to_thread(
            search_service.search, payload.question, top_k
        )
        if not citations:
            raise HTTPException(
                status_code=404,
                detail="Nenhum documento relevante encontrado no banco vetorial.",
            )
        answer, llm_ms = await generate_answer(payload.question, citations)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar pergunta: {exc}",
        ) from exc

    total_ms = round((time.perf_counter() - total_start) * 1000, 2)
    return AnswerResponse(
        answer=answer,
        citations=citations,
        timings=Timings(
            embedding_ms=embedding_ms,
            search_ms=search_ms,
            llm_ms=llm_ms,
            total_ms=total_ms,
        ),
        model=settings.ollama_model,
    )
