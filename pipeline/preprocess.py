"""Pré-processamento em lotes de arquivos JSONL para a indexação semântica.

Os chunks são definidos em TOKENS do tokenizer do modelo de embeddings, e não em
palavras. O padrão de 100 tokens deixa margem para os tokens especiais de modelos
com limite de sequência de 128 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
load_dotenv()


class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str: ...


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def clean_text(value: Any) -> str:
    """Normaliza espaços e remove caracteres de controle não imprimíveis."""
    if is_missing(value):
        return ""
    text = str(value)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def first_nonempty(*values: Any, default: str = "") -> str:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return default


def chunk_text(
    text: str,
    tokenizer: TokenizerProtocol,
    max_tokens: int = 100,
    overlap_tokens: int = 15,
) -> list[str]:
    """Divide um texto em chunks respeitando o número de tokens do modelo."""
    if max_tokens < 1:
        raise ValueError("max_tokens deve ser maior que zero")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens deve ser >= 0 e menor que max_tokens")

    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    model_max_length = getattr(tokenizer, "model_max_length", None)
    special_tokens = (
        tokenizer.num_special_tokens_to_add(pair=False)
        if hasattr(tokenizer, "num_special_tokens_to_add")
        else 0
    )
    if isinstance(model_max_length, int) and 0 < model_max_length < 1_000_000:
        if max_tokens + special_tokens > model_max_length:
            raise ValueError(
                f"max_tokens={max_tokens} + {special_tokens} tokens especiais excede "
                f"o limite do tokenizer ({model_max_length})."
            )

    step = max_tokens - overlap_tokens
    chunks: list[str] = []
    for start in range(0, len(token_ids), step):
        token_slice = token_ids[start : start + max_tokens]
        decoded = tokenizer.decode(
            token_slice,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if decoded:
            chunks.append(decoded)
        if start + max_tokens >= len(token_ids):
            break
    return chunks


def stable_doc_id(record: dict[str, Any], text: str) -> str:
    """Gera identificador reproduzível quando a fonte não fornece um ID."""
    supplied = first_nonempty(record.get("doc_id"), record.get("id"))
    if supplied:
        return supplied
    base = f"{record.get('title', '')}\n{text}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()[:24]


def process_jsonl(
    input_path: Path,
    output_path: Path,
    tokenizer: TokenizerProtocol,
    max_tokens: int = 100,
    overlap_tokens: int = 15,
    batch_docs: int = 2000,
) -> dict[str, int | float]:
    """Lê o JSONL em blocos e grava cada chunk imediatamente no arquivo de saída."""
    if batch_docs < 1:
        raise ValueError("batch_docs deve ser maior que zero")

    started = time.perf_counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    chunk_count = 0
    total_chunk_chars = 0

    # chunksize evita carregar o arquivo inteiro na memória.
    batches = pd.read_json(input_path, lines=True, chunksize=batch_docs, encoding="utf-8", compression="infer")
    with output_path.open("w", encoding="utf-8") as output_file:
        for frame in batches:
            for record in frame.to_dict(orient="records"):
                text = clean_text(record.get("text", ""))
                if len(text) < 30:
                    continue

                document_count += 1
                doc_id = stable_doc_id(record, text)
                chunks = chunk_text(
                    text,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )

                for chunk_index, chunk in enumerate(chunks):
                    output_record = {
                        "doc_id": doc_id,
                        "chunk_id": chunk_index,
                        "title": first_nonempty(record.get("title"), default="Sem título"),
                        "source": first_nonempty(record.get("source"), default="Base Local"),
                        "source_url": first_nonempty(
                            record.get("source_url"), record.get("url")
                        ) or None,
                        "text": chunk,
                    }
                    output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                    chunk_count += 1
                    total_chunk_chars += len(chunk)

    elapsed = time.perf_counter() - started
    average_chars = total_chunk_chars / chunk_count if chunk_count else 0.0
    summary: dict[str, int | float] = {
        "documents": document_count,
        "chunks": chunk_count,
        "average_chunk_characters": round(average_chars, 2),
        "elapsed_seconds": round(elapsed, 2),
    }
    print(
        f"[OK] {document_count} documentos -> {chunk_count} chunks | "
        f"média de {average_chars:.1f} caracteres | {elapsed:.2f}s"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pré-processa JSONL em lotes e cria chunks pelo tokenizer do embedding."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL), help="Modelo/tokenizer usado na indexação")
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("CHUNK_MAX_TOKENS", "100")), help="Tokens de conteúdo por chunk")
    parser.add_argument("--overlap-tokens", type=int, default=int(os.getenv("CHUNK_OVERLAP_TOKENS", "15")), help="Tokens de sobreposição")
    parser.add_argument("--batch-docs", type=int, default=2000, help="Documentos lidos por lote")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Arquivo de entrada não encontrado: {args.input}")

    print(f"[INFO] Carregando tokenizer: {args.model}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    process_jsonl(
        args.input,
        args.output,
        tokenizer,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        batch_docs=args.batch_docs,
    )


if __name__ == "__main__":
    main()
