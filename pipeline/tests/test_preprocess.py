from __future__ import annotations

import json
from pathlib import Path

from pipeline.preprocess import chunk_text, clean_text, process_jsonl, stable_doc_id


class WhitespaceTokenizer:
    """Tokenizer mínimo de teste: um token por palavra separada por espaço."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(
        self,
        token_ids: list[str],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        return " ".join(token_ids)


def test_clean_text_normalizes_whitespace_and_controls() -> None:
    assert clean_text("  História\n\tdo\x00 Brasil  ") == "História do Brasil"


def test_chunk_text_respects_token_limit_and_overlap() -> None:
    tokenizer = WhitespaceTokenizer()
    chunks = chunk_text(
        "um dois três quatro cinco seis sete",
        tokenizer,
        max_tokens=4,
        overlap_tokens=1,
    )
    assert chunks == ["um dois três quatro", "quatro cinco seis sete"]
    assert all(len(tokenizer.encode(chunk)) <= 4 for chunk in chunks)


def test_chunk_text_rejects_invalid_overlap() -> None:
    tokenizer = WhitespaceTokenizer()
    try:
        chunk_text("texto de teste", tokenizer, max_tokens=4, overlap_tokens=4)
    except ValueError:
        pass
    else:
        raise AssertionError("Esperava ValueError para overlap >= max_tokens")


def test_stable_doc_id_is_reproducible() -> None:
    record = {"title": "Título"}
    assert stable_doc_id(record, "texto") == stable_doc_id(record, "texto")


def test_chunk_text_empty_input() -> None:
    assert chunk_text("", WhitespaceTokenizer()) == []


def test_process_jsonl_writes_metadata_and_chunks_by_batches(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "processed" / "chunks.jsonl"
    records = [
        {
            "doc_id": "doc-1",
            "title": "Independência",
            "source": "Amostra",
            "source_url": "https://example.org/doc-1",
            "text": "um dois três quatro cinco seis sete oito nove dez onze doze",
        }
    ]
    input_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    summary = process_jsonl(
        input_path,
        output_path,
        WhitespaceTokenizer(),
        max_tokens=5,
        overlap_tokens=1,
        batch_docs=1,
    )

    chunks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary["documents"] == 1
    assert summary["chunks"] == 3
    assert chunks[0]["doc_id"] == "doc-1"
    assert chunks[0]["source_url"] == "https://example.org/doc-1"
    assert all(len(chunk["text"].split()) <= 5 for chunk in chunks)
