"""Calcula o volume de texto UTF-8 efetivo no corpus, sem contar metadados JSONL."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import TextIO

GIB = 1024**3


def open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def count_text_bytes(path: Path, mode: str = "jsonl", field: str = "text") -> int:
    total = 0
    with open_text(path) as file:
        if mode == "text":
            for line in file:
                total += len(line.encode("utf-8"))
        else:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL inválido em {path}, linha {line_number}: {exc}") from exc
                value = record.get(field, "")
                if value is not None:
                    total += len(str(value).encode("utf-8"))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Mede bytes do conteúdo textual do corpus.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Arquivos .jsonl[.gz] ou texto puro")
    parser.add_argument("--mode", choices=["jsonl", "text"], default="jsonl")
    parser.add_argument("--field", default="text", help="Campo com o texto nos registros JSONL")
    args = parser.parse_args()

    total_bytes = 0
    for path in args.inputs:
        if not path.is_file():
            parser.error(f"Arquivo não encontrado: {path}")
        size = count_text_bytes(path, mode=args.mode, field=args.field)
        total_bytes += size
        print(f"{path}: {size:,} bytes ({size / GIB:.4f} GiB de texto)")

    print(f"TOTAL: {total_bytes:,} bytes ({total_bytes / GIB:.4f} GiB de texto)")
    if total_bytes >= 10 * GIB:
        print("[OK] O conteúdo textual atingiu pelo menos 10 GiB.")
    else:
        print(f"[PENDENTE] Faltam {(10 * GIB - total_bytes) / GIB:.4f} GiB de texto para 10 GiB.")


if __name__ == "__main__":
    main()
