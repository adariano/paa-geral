# PAA RAG local

Protótipo local de busca semântica e geração aumentada por recuperação (RAG)
sobre um piloto de chunks da Wikipédia. A interface prevista nesta entrega é
uma CLI; a resposta pode ser impressa para uma pessoa ou emitida como JSON
para consumo futuro por uma interface gráfica.

## Requisitos

- Python 3.12.
- Ollama instalado e em execução.
- Modelo local, por exemplo `qwen2.5:3b`.
- Artefatos de embeddings para a amostra ou para o piloto.
- Acesso à internet na primeira execução da E5, para baixar os pesos se não
estiverem no cache local.

Instalação de dependências em um ambiente virtual Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-runtime.txt
```

Para avaliação RAGAS, instale também:

```bash
.venv/bin/python -m pip install -r requirements-evaluation.txt
```

Inicie o serviço Ollama se ainda não estiver em execução e baixe o modelo:

```bash
ollama pull qwen2.5:3b
```

O Ollama costuma ser iniciado como serviço pelo instalador. Se a porta local
11434 já estiver ocupada, o serviço provavelmente já está ativo; não inicie
uma segunda instância.

## Fazer uma pergunta

Execute a partir da raiz do projeto:

```bash
python -m paa_rag ask \
  --embedding e5 \
  --search hnsw \
  --question "O que é astronomia?" \
  --k 5 \
  --llm qwen2.5:3b
```

Para obter JSON na saída padrão:

```bash
python -m paa_rag ask \
  --embedding e5 \
  --search hnsw \
  --question "O que é astronomia?" \
  --format json \
  --output evaluation-pilot/answer.json
```

O JSON contém pergunta, resposta, método vetorial, chunks recuperados, citações
com título e URL, e tempos de inicialização, embedding, busca, geração e total.
As mensagens de progresso são enviadas para `stderr`, separadas da saída JSON.

## Gerar embeddings a partir dos dados existentes

Os geradores recebem um JSONL de chunks: um objeto JSON por linha, com pelo
menos `chunk_id` e `text`. Para que as citações incluam título e URL, o objeto
também deve ter `document_id`, `source`, `language`, `title`, `url`,
`chunk_index` e `text_hash`.

### E5 Sentence Embeddings

Com o ambiente Python 3.12 e o piloto existente:

```bash
./.venv-py312/bin/python generate_pilot_embeddings.py \
  --chunks corpus-pilot/chunks.jsonl \
  --output embeddings-pilot/e5-small \
  --model intfloat/multilingual-e5-small \
  --batch-size 32 \
  --limit 100000
```

O gerador codifica os chunks com `passage: ` e cria `vectors.f32.npy` (384
dimensões float32 normalizadas), `chunk_ids.jsonl`, `metadata.jsonl` e
`manifest.json`. Na primeira execução, Sentence Transformers pode baixar os
pesos do Hugging Face.

### Word2Vec Average

O Word2Vec é treinado sobre os textos selecionados; depois, cada chunk é
representado pela média dos vetores das palavras conhecidas:

```bash
./.venv-py312/bin/python generate_word2vec_embeddings.py \
  --chunks corpus-pilot/chunks.jsonl \
  --output embeddings-pilot/word2vec \
  --limit 100000 \
  --dimension 100 \
  --window 5 \
  --min-count 5 \
  --epochs 5 \
  --workers 12
```

O resultado contém os vetores, IDs, metadados, manifesto e `word2vec.model`.
Ao contrário da E5, o Word2Vec desse comando é treinado no conjunto de chunks.

### Apontar para dados e embeddings externos

Os arquivos não precisam estar dentro do diretório de exportação. Informe o
caminho de chunks e de saída ao gerador; depois, passe o manifesto resultante
ao CLI:

```bash
./.venv-py312/bin/python generate_pilot_embeddings.py \
  --chunks /dados/corpus-pilot/chunks.jsonl \
  --output /dados/embeddings/e5-small \
  --limit 100000

python -m paa_rag ask \
  --manifest /dados/embeddings/e5-small/manifest.json \
  --search hnsw \
  --question "O que é astronomia?"
```

Para Word2Vec, use `generate_word2vec_embeddings.py` com os mesmos caminhos.
Mantenha juntos o manifesto, os vetores, IDs e metadados; Word2Vec também exige
seu arquivo de modelo. O campo `chunks` do manifesto deve apontar para o JSONL
correspondente, pois o CLI o utiliza para carregar os textos recuperados.

O parâmetro `--limit 100000` limita o piloto. `--limit 0` processaria todos os
registros do arquivo; **não use isso no corpus completo nesta máquina**. E5
precisaria fazer inferência para cada chunk; o gerador Word2Vec mantém o corpus
selecionado em memória durante o treinamento. Os resultados medidos neste
projeto são apenas para 100.000 chunks.

Para verificar uma geração, confira `count` e `dimension` no manifesto e
compare `count` com o número de linhas de `chunk_ids.jsonl`. O manifesto e os
três arquivos de saída precisam pertencer à mesma geração.

Para perguntas sucessivas sem recarregar o modelo e o índice:

```bash
python -m paa_rag chat --embedding e5 --search hnsw
```

Digite `/sair` para encerrar.

## Selecionar os quatro métodos

| Representação (`--embedding`) | Busca (`--search`) | Manifesto padrão |
|---|---|---|
| `e5` | `cosine_exact` | `embeddings-pilot/e5-small/manifest.json` |
| `e5` | `hnsw` | `embeddings-pilot/e5-small/manifest.json` |
| `word2vec` | `cosine_exact` | `embeddings-pilot/word2vec/manifest.json` |
| `word2vec` | `hnsw` | `embeddings-pilot/word2vec/manifest.json` |

Exemplo Word2Vec + cosseno:

```bash
python -m paa_rag ask \
  --embedding word2vec \
  --search cosine_exact \
  --question "O que é astronomia?"
```

O manifesto pode ser sobrescrito com `--manifest`. `--k` define quantos chunks
recuperar. `--ef` configura o parâmetro de busca HNSW e não altera o cosseno
exato. `--temperature` controla a aleatoriedade da geração.

## Avaliar um conjunto de perguntas

```bash
python -m paa_rag evaluate \
  --embedding e5 \
  --search hnsw \
  --dataset evaluation-pilot/questions-pt-rag.jsonl \
  --limit 50 \
  --output evaluation-pilot/rag-pilot-report.json
```

O relatório gerado pode ser avaliado pelo RAGAS:

```bash
python evaluate_ragas.py \
  --rag-report evaluation-pilot/rag-pilot-report.json \
  --evaluator-model qwen2.5:3b \
  --output evaluation-pilot/ragas-report.json
```

As respostas de referência ficam em
`evaluation-pilot/expected-answers-pt.jsonl`; são concisas e usadas como
referência factual. Revise-as antes de tratar os escores automáticos como
avaliação definitiva.

## Dados e limitações

- O corpus completo tem 7.082.981 chunks e 9.075.006.885 bytes de texto:
  aproximadamente 8,45 GiB ou 9,08 GB decimais.
- O requisito do enunciado é pelo menos 10 GB de texto puro. O corpus existente
  fica abaixo desse mínimo; a limitação e os desafios para ampliar e processar
  o volume estão registrados em `docs/PIPELINE.md`.
- Os benchmarks vetoriais foram executados com 100.000 chunks, não com o
  corpus completo. Esse limite evita exigir vários dias de inferência E5 no
  computador disponível e permite comparar métodos com recursos locais.
- Não foram gerados embeddings para os 7 milhões de chunks.
- A amostra de exportação é menor que 100 MB e serve para smoke test. Ela não
  reproduz a representatividade nem os resultados do piloto integral.

## Resultados de recuperação já medidos

Em 50 perguntas em português, E5 obteve Recall@10 de 1,00 e MRR/MAP de
0,8065; Word2Vec Average obteve Recall@10 de 0,30 e MRR/MAP de 0,1319. No
piloto medido, HNSW teve latência média de 2,49 ms contra 9,04 ms para o
cosseno exato com E5, sem diferença nas métricas de ranking reportadas.
Consulte os JSON em `embeddings-pilot/*/retrieval-report.json`. Esses números
são resultados de recuperação, não resultados RAGAS nem uma avaliação de
qualidade da resposta da LLM.

## Estrutura

- `paa_rag/`: engine reutilizável e comandos `ask`, `chat` e `evaluate`.
- `generate_pilot_embeddings.py`: embeddings E5.
- `generate_word2vec_embeddings.py`: treinamento Word2Vec e média vetorial.
- `benchmark_retrieval.py`: Recall@k, MRR, MAP e latência de recuperação.
- `rag_pilot.py`: compatibilidade com comandos de avaliação anteriores.
- `evaluate_ragas.py`: avaliação das respostas com RAGAS e Ollama.
- `tests/`: testes unitários de busca exata.
- `docs/PIPELINE.md`: arquitetura, métricas, complexidade e terminologia.
- `build_export.py` (na árvore de desenvolvimento): cria uma cópia portátil
  sem alterar os dados originais; não é necessário no diretório exportado.

Consulte `docs/PIPELINE.md` para detalhes de pipeline, parâmetros, bibliotecas
e licenças.
