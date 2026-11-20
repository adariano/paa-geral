# Metodologia de benchmark

## Métodos comparados

1. **TF-IDF + cosseno (busca exata):** representa os textos por vetores esparsos e calcula os scores de todos os documentos.
2. **Word2Vec Average:** treina vetores de palavras com Gensim e representa cada documento pela média normalizada dos vetores de suas palavras.
3. **Sentence Embeddings + busca exata:** codifica cada chunk com Sentence Transformers e compara a consulta com todos os vetores, usando produto interno em vetores normalizados (equivalente ao cosseno).
4. **HNSW:** constrói um índice aproximado sobre os mesmos Sentence Embeddings. A recuperação exata é a referência para `Recall@K`.

## Métricas registradas

- `index_build_ms`: tempo de construção/representação do método. Para HNSW, mede a construção do grafo e não inclui o tempo de geração dos Sentence Embeddings, que é apresentado na linha do baseline de embeddings.
- `avg_query_ms` e `p95_query_ms`: latência de consulta, incluindo a codificação da pergunta quando aplicável.
- `index_memory_mb_estimate`: estimativa do espaço da representação/índice; para HNSW inclui uma aproximação do espaço das conexões.
- `process_rss_mb_after`: memória residente do processo após a etapa, útil como observação, não como medição isolada do índice.
- `recall_vs_sentence_exact_at_k`: proporção média dos vizinhos HNSW que também aparecem no top-k exato de Sentence Embeddings. Não substitui um conjunto de relevância anotado por humanos.

## Complexidades: como apresentar

Seja `N` o número de documentos, `d` a dimensão do vetor denso, `V` o vocabulário e `nnz(q)` a quantidade de termos não nulos na consulta.

| Método | Consulta (modelo simplificado) | Observação |
|---|---|---|
| TF-IDF + cosseno | Aproximadamente `O(N * nnz(q))`, mais seleção top-k | Matriz esparsa; custo depende da distribuição de termos |
| Word2Vec Average | Vetorização da consulta + `O(N * d)` | Treinamento do Word2Vec é uma etapa adicional |
| Sentence Embeddings exato | Codificação da pergunta + `O(N * d)` | É baseline de recuperação exata |
| HNSW | Dependente do grafo e dos parâmetros | Busca aproximada; não prometa garantia universal `O(log N)` |

A construção do HNSW também depende de `N`, `M`, `ef_construction` e da distribuição dos dados. `ef_search` influencia o compromisso entre latência e recall.

## Protocolo experimental recomendado

- Fixar a mesma máquina, sistema, versão de Python, modelo e conjunto de perguntas para as comparações.
- Registrar CPU, RAM, GPU (se houver), sistema operacional e versão/quantização da LLM nos resultados.
- Executar mais de uma repetição (`--query-repeats`) e reportar média e P95; descartar ou identificar a primeira execução fria, se houver carregamento de modelos.
- Comparar tamanhos crescentes de corpus e não extrapolar um resultado de 10 mil documentos para 10 GB sem medição.
- Usar `Recall@K` de HNSW contra o baseline exato e, em separado, avaliar relevância real com perguntas e documentos julgados.
- Não misturar o tempo de inferência da LLM com a latência da busca vetorial; a API mede embedding, busca e LLM separadamente.

O script padrão limita a amostra a 10.000 chunks para evitar estouro de memória. Para o corpus integral, planejem experimentos por faixas de tamanho e relatem claramente o limite de memória.
