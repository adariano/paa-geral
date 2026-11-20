"""Gera embeddings e envia chunks ao Qdrant em lotes limitados em memória."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from itertools import islice
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()


def iter_jsonl_batches(path: Path, batch_size: int) -> Iterator[list[dict]]:
    """Produz lotes de registros sem carregar o arquivo inteiro."""
    line_number = 0
    with path.open("r", encoding="utf-8") as file:
        while True:
            lines = list(islice(file, batch_size))
            if not lines:
                break
            batch = []
            for line in lines:
                line_number += 1
                if not line.strip():
                    continue
                try:
                    batch.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido na linha {line_number}: {exc}") from exc
            if batch:
                yield batch


def point_id(record: dict) -> str:
    """ID determinístico: permite reexecutar a ingestão sem duplicar pontos."""
    doc_id = str(record.get("doc_id", "unknown"))
    chunk_id = str(record.get("chunk_id", 0))
    source_identity = str(record.get("source_url") or record.get("source") or "")
    identity = f"paa-historia:{source_identity}:{doc_id}:{chunk_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def main() -> None:
    parser = argparse.ArgumentParser(description="Indexa JSONL no Qdrant com processamento em lotes.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64, help="Pontos enviados por requisição")
    parser.add_argument("--encoder-batch-size", type=int, default=32, help="Textos codificados por lote")
    parser.add_argument("--recreate", action="store_true", help="Apaga e recria a coleção")
    parser.add_argument("--max-records", type=int, default=0, help="Limite opcional para testes; 0 = todos")
    args = parser.parse_args()

    if args.batch_size < 1 or args.encoder_batch_size < 1:
        parser.error("batch-size e encoder-batch-size devem ser maiores que zero")
    if args.max_records < 0:
        parser.error("max-records deve ser >= 0")
    if not args.input.exists():
        parser.error(f"Arquivo de entrada não encontrado: {args.input}")

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "historia_brasil")
    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    started = time.perf_counter()
    print(f"[INFO] Carregando modelo de embeddings: {model_name}")
    encoder = SentenceTransformer(model_name)
    vector_size = encoder.get_sentence_embedding_dimension()

    client = QdrantClient(host=host, port=port)
    exists = client.collection_exists(collection)
    if exists and args.recreate:
        client.delete_collection(collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"[INFO] Coleção '{collection}' criada (dimensão={vector_size}).")
    else:
        collection_info = client.get_collection(collection)
        vectors_config = collection_info.config.params.vectors
        if hasattr(vectors_config, "size") and vectors_config.size != vector_size:
            raise RuntimeError(
                f"A coleção existente tem dimensão {vectors_config.size}, mas o modelo "
                f"{model_name} produz vetores de dimensão {vector_size}. Use --recreate "
                "(isso apaga os pontos existentes) ou escolha outra coleção."
            )
        # Detecta mistura acidental de modelos com a mesma dimensão quando há pontos.
        sample_points, _ = client.scroll(
            collection_name=collection,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if sample_points:
            previous_model = (sample_points[0].payload or {}).get("_embedding_model")
            if previous_model and previous_model != model_name:
                raise RuntimeError(
                    f"A coleção já contém embeddings do modelo '{previous_model}', "
                    f"mas foi solicitado '{model_name}'. Use --recreate ou outra coleção."
                )
            if not previous_model:
                print(
                    "[AVISO] A coleção contém pontos sem marcador de modelo de embedding. "
                    "Confirme a compatibilidade antes de misturar novos vetores."
                )

    total_points = 0
    progress = tqdm(desc="Indexando lotes", unit="lote")
    for records in iter_jsonl_batches(args.input, args.batch_size):
        if args.max_records:
            remaining = args.max_records - total_points
            if remaining <= 0:
                break
            records = records[:remaining]

        texts = [str(record.get("text", "")) for record in records]
        if any(not text.strip() for text in texts):
            raise ValueError("Há chunks vazios no JSONL; execute o pré-processamento novamente.")

        vectors = encoder.encode(
            texts,
            batch_size=args.encoder_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        points = []
        for record, vector in zip(records, vectors):
            payload = {**record, "_embedding_model": model_name}
            points.append(
                PointStruct(
                    id=point_id(record),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)
        total_points += len(points)
        progress.update(1)
        progress.set_postfix(pontos=total_points)
        if args.max_records and total_points >= args.max_records:
            break

    progress.close()
    elapsed = time.perf_counter() - started
    print(
        f"[OK] {total_points} pontos enviados em {elapsed:.2f}s "
        f"para a coleção '{collection}'."
    )


if __name__ == "__main__":
    main()
