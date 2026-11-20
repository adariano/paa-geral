import time

import httpx

from backend.app.config import settings
from backend.app.schemas import Citation


def build_prompt(question: str, citations: list[Citation]) -> str:
    context_blocks = []
    for idx, citation in enumerate(citations, start=1):
        url_line = f"\nURL: {citation.source_url}" if citation.source_url else ""
        context_blocks.append(
            f"[Fonte {idx} - {citation.document}; origem: {citation.source}]"
            f"{url_line}\n{citation.snippet}"
        )
    context_text = "\n\n".join(context_blocks)

    return (
        "Você é um assistente especializado em História Brasileira.\n"
        "Responda em português, de forma clara e objetiva, usando estritamente o contexto.\n"
        "Indique entre colchetes as fontes que sustentam cada afirmação (ex.: [Fonte 1]).\n"
        "Não invente datas, nomes, fatos ou referências. Se o contexto for insuficiente, diga isso.\n\n"
        f"CONTEXTO RECUPERADO:\n{context_text}\n\n"
        f"PERGUNTA:\n{question}\n\n"
        "RESPOSTA:"
    )


async def generate_answer(question: str, citations: list[Citation]) -> tuple[str, float]:
    prompt = build_prompt(question, citations)
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    llm_ms = (time.perf_counter() - started) * 1000
    answer = data.get("response", "").strip()
    if not answer:
        raise RuntimeError("O Ollama retornou uma resposta vazia.")
    return answer, round(llm_ms, 2)
