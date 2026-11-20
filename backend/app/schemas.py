from pydantic import BaseModel, Field, HttpUrl


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Pergunta em linguagem natural sobre História Brasileira.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Quantidade de trechos recuperados na busca semântica.",
    )


class Citation(BaseModel):
    document: str
    source: str
    source_url: HttpUrl | None = None
    doc_id: str | None = None
    snippet: str
    score: float


class Timings(BaseModel):
    embedding_ms: float
    search_ms: float
    llm_ms: float
    total_ms: float


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
    timings: Timings
    model: str
