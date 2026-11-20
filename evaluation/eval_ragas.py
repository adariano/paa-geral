"""Avalia respostas do endpoint com métricas de qualidade RAGAS e latência.

O conjunto atual contém poucas perguntas e funciona como smoke test. Para os
resultados acadêmicos, ampliem-no e revisem manualmente as respostas de referência.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import httpx
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

load_dotenv()

API_URL = os.getenv("EVAL_API_URL", "http://localhost:8000/api/ask")
OLLAMA_MODEL = os.getenv("RAGAS_LLM_MODEL", os.getenv("OLLAMA_MODEL", "phi3:mini"))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
TOP_K = int(os.getenv("EVAL_TOP_K", "4"))


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return float(pd.Series(values).quantile(value))


def main() -> None:
    questions_path = Path(os.getenv("EVAL_QUESTIONS_FILE", "evaluation/test_questions.json"))
    if not questions_path.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {questions_path}")

    test_cases = json.loads(questions_path.read_text(encoding="utf-8"))
    if not test_cases:
        raise ValueError("O conjunto de avaliação está vazio.")

    rows: list[dict[str, object]] = []
    latencies: list[float] = []

    with httpx.Client(timeout=240.0) as client:
        for index, item in enumerate(test_cases, start=1):
            question = str(item["question"])
            response = client.post(API_URL, json={"question": question, "top_k": TOP_K})
            response.raise_for_status()
            data = response.json()
            contexts = [
                str(citation.get("snippet", ""))
                for citation in data.get("citations", [])
                if citation.get("snippet")
            ]
            rows.append({
                "question": question,
                "answer": data["answer"],
                "contexts": contexts,
                "ground_truth": str(item["ground_truth"]),
            })
            latencies.append(float(data["timings"]["total_ms"]))
            print(f"[{index}/{len(test_cases)}] Resposta coletada")

    dataset = Dataset.from_list(rows)
    evaluator_llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = result.to_pandas()
    scores["total_ms"] = latencies
    scores.to_csv(output_dir / "ragas_results.csv", index=False)

    latency_summary = {
        "num_questions": len(latencies),
        "avg_total_ms": statistics.mean(latencies),
        "p50_total_ms": percentile(latencies, 0.50),
        "p95_total_ms": percentile(latencies, 0.95),
        "evaluator_model": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
        "warning": (
            "A avaliação depende da qualidade do conjunto de referência e do modelo avaliador. "
            "Revise amostras manualmente e não interprete métricas automáticas como verdade absoluta."
        ),
    }
    (output_dir / "latency_summary.json").write_text(
        json.dumps(latency_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(scores)
    print(f"\n[OK] Resultados em {output_dir.resolve()}")


if __name__ == "__main__":
    main()
