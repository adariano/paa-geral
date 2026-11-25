import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pilot results with RAGAS.",
    )
    parser.add_argument(
        "--rag-report",
        default="evaluation-pilot/rag-pilot-report.json",
    )
    parser.add_argument(
        "--output",
        default="evaluation-pilot/ragas-report.json",
    )
    parser.add_argument("--evaluator-model", default="qwen2.5:3b")
    parser.add_argument("--evaluator-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as error:
        raise SystemExit(
            "Missing RAGAS dependencies. Install ragas, datasets and "
            f"langchain-ollama. Details: {error}"
        )

    report = json.loads(Path(args.rag_report).read_text(encoding="utf-8"))
    rows = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for item in report["results"]:
        rows["user_input"].append(item["question"])
        rows["response"].append(item["answer"])
        rows["retrieved_contexts"].append(item["contexts"])
        rows["reference"].append(item["ground_truth"])

    evaluator_llm = ChatOllama(
        model=args.evaluator_model,
        base_url=args.evaluator_url,
        temperature=0.0,
    )
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    ]
    ragas_result = evaluate(
        Dataset.from_dict(rows),
        metrics=metrics,
        llm=evaluator_llm,
    )

    score_rows = ragas_result.scores
    metric_names = list(score_rows[0]) if score_rows else []
    summary = {}
    per_question = []
    for metric_name in metric_names:
        values = [
            row[metric_name]
            for row in score_rows
            if row.get(metric_name) is not None
            and not math.isnan(float(row[metric_name]))
        ]
        summary[metric_name] = (
            sum(float(value) for value in values) / len(values)
            if values
            else None
        )
    for row in score_rows:
        per_question.append({
            name: (
                float(value)
                if value is not None and not math.isnan(float(value))
                else None
            )
            for name, value in row.items()
        })

    output = {
        "evaluator_model": args.evaluator_model,
        "questions": len(rows["user_input"]),
        "ragas": per_question,
        "summary": summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
