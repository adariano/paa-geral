"""Benchmark de recuperação: TF-IDF/cosseno, Word2Vec Average, embeddings exatos e HNSW.

O benchmark limita a amostra por padrão para evitar tentar carregar o corpus inteiro
em RAM. O valor de max-docs usado é sempre registrado no CSV. HNSW é aproximado: sua
latência típica pode crescer de forma sublinear, mas não há garantia universal O(log N).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import hnswlib
import numpy as np
import pandas as pd
import psutil
from dotenv import load_dotenv
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def load_texts(path: Path, max_docs: int = 10_000) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if text:
                texts.append(text)
            if max_docs > 0 and len(texts) >= max_docs:
                break
    return texts


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Seleciona top-k sem ordenar todo o vetor de N scores."""
    count = min(top_k, scores.size)
    if count == 0:
        return np.array([], dtype=np.int64)
    if count == scores.size:
        candidates = np.arange(scores.size)
    else:
        candidates = np.argpartition(scores, scores.size - count)[-count:]
    return candidates[np.argsort(scores[candidates])[::-1]]


def latency_summary(samples_ms: list[float]) -> tuple[float, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return float(np.mean(values)), float(np.percentile(values, 95))


def run_benchmark(
    texts: list[str],
    queries: list[str],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
    hnsw_m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
    word2vec_epochs: int = 10,
    encoder_batch_size: int = 64,
    query_repeats: int = 3,
) -> pd.DataFrame:
    if not texts:
        raise ValueError("Nenhum texto válido foi encontrado no arquivo de entrada.")
    if not queries:
        raise ValueError("Informe pelo menos uma consulta para o benchmark.")
    if query_repeats < 1:
        raise ValueError("query_repeats deve ser maior que zero")

    results: list[dict[str, object]] = []
    n_docs = len(texts)
    process = psutil.Process()

    def append_result(
        method: str,
        build_ms: float,
        query_samples: list[float],
        index_bytes: int,
        complexity: str,
        recall: float | None = None,
    ) -> None:
        mean_ms, p95_ms = latency_summary(query_samples)
        results.append({
            "method": method,
            "num_docs": n_docs,
            "num_queries": len(queries),
            "index_build_ms": round(build_ms, 2),
            "avg_query_ms": round(mean_ms, 4),
            "p95_query_ms": round(p95_ms, 4),
            "index_memory_mb_estimate": round(index_bytes / (1024**2), 3),
            "process_rss_mb_after": round(process.memory_info().rss / (1024**2), 2),
            "recall_vs_sentence_exact_at_k": None if recall is None else round(recall, 4),
            "top_k": top_k,
            "complexity_note": complexity,
        })

    # 1. TF-IDF + cosseno exato. A matriz é esparsa.
    started = time.perf_counter()
    tfidf = TfidfVectorizer(dtype=np.float32, max_features=500_000)
    doc_matrix = tfidf.fit_transform(texts)
    tfidf_build_ms = (time.perf_counter() - started) * 1000
    tfidf_bytes = doc_matrix.data.nbytes + doc_matrix.indices.nbytes + doc_matrix.indptr.nbytes
    tfidf_bytes += sum(len(token.encode("utf-8")) + 32 for token in tfidf.vocabulary_)
    tfidf_query_ms: list[float] = []
    for _ in range(query_repeats):
        for query in queries:
            query_started = time.perf_counter()
            query_vector = tfidf.transform([query])
            scores = (query_vector @ doc_matrix.T).toarray().ravel()
            _ = topk_indices(scores, top_k)
            tfidf_query_ms.append((time.perf_counter() - query_started) * 1000)
    append_result(
        "TF-IDF + similaridade de cosseno (exata)",
        tfidf_build_ms,
        tfidf_query_ms,
        tfidf_bytes,
        "Construção depende dos tokens; consulta O(N * nnz(q)) + seleção top-k O(N) esperada.",
    )
    del tfidf, doc_matrix

    # 2. Word2Vec Average: treina vetores de palavra e calcula a média por documento.
    tokenized_docs = [tokenize(text) for text in texts]
    started = time.perf_counter()
    w2v = Word2Vec(
        sentences=tokenized_docs,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        seed=42,
        epochs=word2vec_epochs,
    )

    def embed_w2v(tokens: list[str]) -> np.ndarray:
        vectors = [w2v.wv[token] for token in tokens if token in w2v.wv]
        if not vectors:
            return np.zeros(w2v.vector_size, dtype=np.float32)
        vector = np.mean(vectors, axis=0).astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    w2v_matrix = np.vstack([embed_w2v(tokens) for tokens in tokenized_docs])
    w2v_build_ms = (time.perf_counter() - started) * 1000
    w2v_query_ms: list[float] = []
    for _ in range(query_repeats):
        for query in queries:
            query_started = time.perf_counter()
            query_vector = embed_w2v(tokenize(query))
            scores = w2v_matrix @ query_vector
            _ = topk_indices(scores, top_k)
            w2v_query_ms.append((time.perf_counter() - query_started) * 1000)
    w2v_bytes = w2v_matrix.nbytes + sum(
        len(word.encode("utf-8")) + w2v.vector_size * 4 for word in w2v.wv.index_to_key
    )
    append_result(
        "Word2Vec Average (Gensim)",
        w2v_build_ms,
        w2v_query_ms,
        w2v_bytes,
        "Treino depende do total de tokens/épocas; consulta exata O(N * d) + top-k.",
    )
    del w2v_matrix, w2v, tokenized_docs

    # 3. Sentence embeddings + busca exata (baseline de referência para recall do HNSW).
    encoder = SentenceTransformer(model_name)
    started = time.perf_counter()
    embedding_matrix = encoder.encode(
        texts,
        batch_size=encoder_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    embeddings_build_ms = (time.perf_counter() - started) * 1000
    exact_query_ms: list[float] = []
    exact_neighbors: list[np.ndarray] = []
    for repeat in range(query_repeats):
        for query_index, query in enumerate(queries):
            query_started = time.perf_counter()
            query_vector = encoder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32, copy=False)
            scores = embedding_matrix @ query_vector
            neighbors = topk_indices(scores, top_k)
            if repeat == 0:
                exact_neighbors.append(neighbors)
            exact_query_ms.append((time.perf_counter() - query_started) * 1000)
    append_result(
        "Sentence Embeddings + busca exata por cosseno",
        embeddings_build_ms,
        exact_query_ms,
        embedding_matrix.nbytes,
        "Codificação depende do tamanho dos textos; consulta exata O(N * d) + top-k.",
    )

    # 4. HNSW. A construção do grafo é medida separadamente da geração dos embeddings.
    hnsw_started = time.perf_counter()
    hnsw_index = hnswlib.Index(space="cosine", dim=embedding_matrix.shape[1])
    hnsw_index.init_index(
        max_elements=n_docs,
        ef_construction=ef_construction,
        M=hnsw_m,
        random_seed=42,
    )
    hnsw_index.add_items(embedding_matrix, np.arange(n_docs, dtype=np.int64))
    hnsw_index.set_ef(max(ef_search, top_k))
    hnsw_build_ms = (time.perf_counter() - hnsw_started) * 1000

    hnsw_query_ms: list[float] = []
    recalls: list[float] = []
    for _ in range(query_repeats):
        for query, exact_ids in zip(queries, exact_neighbors):
            query_started = time.perf_counter()
            query_vector = encoder.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32, copy=False)
            approximate_ids, _ = hnsw_index.knn_query(query_vector, k=min(top_k, n_docs))
            hnsw_query_ms.append((time.perf_counter() - query_started) * 1000)
            exact_set = set(int(item) for item in exact_ids)
            approximate_set = set(int(item) for item in approximate_ids[0])
            recalls.append(len(exact_set & approximate_set) / max(1, len(exact_set)))

    hnsw_recall = float(np.mean(recalls)) if recalls else 0.0
    # A estimativa inclui a matriz vetorial e um orçamento aproximado para as conexões.
    hnsw_bytes_estimate = embedding_matrix.nbytes + n_docs * hnsw_m * 2 * 4
    append_result(
        f"HNSW (hnswlib; M={hnsw_m}, efC={ef_construction}, efS={max(ef_search, top_k)})",
        hnsw_build_ms,
        hnsw_query_ms,
        hnsw_bytes_estimate,
        "Busca aproximada; desempenho típico pode ser sublinear, mas não há garantia universal O(log N); depende de M/efSearch.",
        recall=hnsw_recall,
    )

    return pd.DataFrame(results)


def load_queries(path: Path | None) -> list[str]:
    if path is None:
        return [
            "Quando ocorreu a Proclamação da República?",
            "Quem assinou a Lei Áurea?",
            "Qual foi o objetivo da Inconfidência Mineira?",
            "Quais foram as consequências da Independência do Brasil?",
            "Qual foi a importância da Constituição de 1988?",
        ]
    items = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["question"]) for item in items if item.get("question")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara métodos de recuperação semântica.")
    parser.add_argument("--input", type=Path, default=Path("data/sample/sample_historia_brasil.jsonl"))
    parser.add_argument("--queries-file", type=Path, default=Path("evaluation/test_questions.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/metrics.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-docs", type=int, default=10_000, help="Máximo de documentos; 0 = todos")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=50)
    parser.add_argument("--word2vec-epochs", type=int, default=10)
    parser.add_argument("--query-repeats", type=int, default=3)
    args = parser.parse_args()

    if args.max_docs < 0 or args.top_k < 1:
        parser.error("max-docs deve ser >= 0 e top-k deve ser >= 1")
    if args.query_repeats < 1:
        parser.error("query-repeats deve ser >= 1")
    if args.hnsw_m < 2 or args.ef_construction < 1 or args.ef_search < 1:
        parser.error("Parâmetros HNSW inválidos: M >= 2, ef-construction >= 1 e ef-search >= 1")
    if not args.input.exists():
        parser.error(f"Arquivo de entrada não encontrado: {args.input}")
    if args.queries_file and not args.queries_file.exists():
        parser.error(f"Arquivo de perguntas não encontrado: {args.queries_file}")

    texts = load_texts(args.input, max_docs=args.max_docs)
    queries = load_queries(args.queries_file)
    print(f"[INFO] Benchmark com {len(texts)} documentos e {len(queries)} consultas.")
    if args.max_docs == 0:
        print("[AVISO] max-docs=0 carrega todo o arquivo; monitore a memória disponível.")
    elif len(texts) == args.max_docs:
        print(f"[INFO] Amostra limitada a {args.max_docs} documentos; o resultado não representa automaticamente o corpus completo.")

    metrics = run_benchmark(
        texts=texts,
        queries=queries,
        top_k=args.top_k,
        model_name=args.model,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        word2vec_epochs=args.word2vec_epochs,
        encoder_batch_size=args.encoder_batch_size,
        query_repeats=args.query_repeats,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))
    print(f"\n[OK] Resultados salvos em {args.output}")


if __name__ == "__main__":
    main()
