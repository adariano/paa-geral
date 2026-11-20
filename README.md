# PAA — Busca Semântica em História Brasileira

Projeto da disciplina **Projeto e Análise de Algoritmos (PAA 2/2026)**, da Universidade de Brasília. A aplicação permite fazer perguntas em linguagem natural sobre História Brasileira, recuperar trechos de documentos por busca semântica, produzir respostas com uma LLM local e exibir as fontes utilizadas.


## Integrantes

| Nome | Matrícula |
| :--- | :--- |
| Adriano Lopes de Sousa | 170097960 |
| Breno Back dos Santos Miranda da Silva | 190063980 |
| Gabriel Bismarck Gonçalves Fagundes | 170103323 |
| Mariana Soares Oliveira | 23101363 |
| Mário Augusto Vieira dos Santos | 231035778 |

## Stack tecnológica

| Camada | Tecnologia | Uso no projeto |
| :--- | :--- | :--- |
| Front-end | Vue 3, TypeScript, Tailwind CSS | Interface de perguntas, respostas, tempos e fontes |
| Back-end | Python, FastAPI | API e orquestração do fluxo RAG |
| Processamento | Pandas, NumPy | Leitura por lotes e operações numéricas |
| Word embeddings | Gensim / Word2Vec | Vetores de palavras e média por documento |
| Sentence embeddings | Sentence Transformers | Vetorização de perguntas e chunks |
| Busca exata | TF-IDF + cosseno e produto interno normalizado | Baselines lineares |
| Busca aproximada | hnswlib (HNSW) | Comparação de busca aproximada por vizinhos |
| Banco vetorial | Qdrant | Armazenamento e recuperação de vetores com metadados |
| LLM local | Phi-3 Mini ou SmolLM | Geração da resposta com contexto recuperado |
| Execução da LLM | Ollama | Servir o modelo localmente |
| Avaliação | RAGAS | Faithfulness, answer relevancy, context precision e context recall |
| Versionamento | Git e GitHub | Colaboração e histórico do código |

**Nota sobre FAISS:** a dependência FAISS foi retirada desta configuração inicial porque o benchmark entregue utiliza `hnswlib` e a busca exata é feita diretamente com NumPy/scikit-learn. FAISS pode ser acrescentado posteriormente como outro método experimental, com seu próprio teste e registro de versão.

## Estrutura do repositório

```text
paa-historia-brasil/
├── backend/
│   └── app/
│       ├── main.py                 # API FastAPI
│       ├── schemas.py              # Contratos da API e das citações
│       └── services/               # Qdrant e Ollama
├── frontend/                       # Vue + TypeScript + Tailwind
├── pipeline/
│   ├── download.py                 # Download por URL direta
│   ├── report_corpus.py             # Mede somente bytes de texto
│   ├── preprocess.py               # JSONL em lotes e chunks por tokens
│   └── ingest_qdrant.py             # Embeddings e ingestão incremental
├── benchmarks/
│   ├── benchmark_search.py         # TF-IDF, Word2Vec, embeddings e HNSW
│   └── results/                    # Resultados locais (não versionados)
├── evaluation/
│   ├── eval_ragas.py                # Avaliação RAGAS e latência
│   └── test_questions.json         # Conjunto inicial de smoke tests
├── data/
│   ├── sample/                     # Pequena amostra versionada
│   ├── raw/                        # Corpus original (não versionado)
│   └── processed/                  # Chunks JSONL (não versionados)
├── docs/                           # Fontes, licenças e metodologia
├── docker-compose.yml              # Qdrant com versão fixada
├── .env.example
└── .gitignore
```

## Pré-requisitos

- Python **3.11** recomendado (use a mesma versão entre os integrantes).
- Node.js 18 ou superior e npm.
- Docker Desktop ou Docker Engine com Docker Compose.
- Ollama instalado separadamente na máquina que executará a LLM.
- Espaço em disco para o corpus, dados processados e armazenamento Qdrant. O consumo de RAM depende do modelo e da configuração de lotes.

## Configuração local

### 1. Clonar e preparar as variáveis

```bash
git clone https://github.com/masolivs/paa-historia-brasil.git
cd paa-historia-brasil
cp .env.example .env
cp frontend/.env.example frontend/.env
```

No PowerShell, substitua os dois comandos `cp` por:

```powershell
Copy-Item .env.example .env
Copy-Item frontend/.env.example frontend/.env
```

Revise o `.env`. O mesmo `EMBEDDING_MODEL` deve ser utilizado no pré-processamento, na ingestão e na API. Alterar o modelo exige gerar novamente os embeddings e recriar a coleção vetorial.

### 2. Iniciar Qdrant

```bash
docker compose up -d qdrant
```

- API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

A imagem do Qdrant está fixada em `v1.12.1` para reduzir diferenças entre as máquinas do grupo.

### 3. Instalar e iniciar Ollama

Instale o Ollama conforme o sistema operacional e baixe o modelo escolhido pelo grupo. Exemplo:

```bash
ollama pull phi3:mini
ollama list
```

A disponibilidade e o desempenho do modelo dependem da máquina. Registrem nos experimentos o modelo e a quantização efetivamente usados. O Ollama não é iniciado pelo `docker-compose.yml` nesta configuração.

### 4. Preparar o ambiente do pipeline e carregar a amostra

```bash
python -m venv .venv-pipeline
```

Ativação no PowerShell:

```powershell
.\.venv-pipeline\Scripts\Activate.ps1
```

Ativação no Linux/macOS:

```bash
source .venv-pipeline/bin/activate
```

Instale as dependências e execute:

```bash
pip install -r pipeline/requirements.txt
python pipeline/preprocess.py --input data/sample/sample_historia_brasil.jsonl --output data/processed/sample_chunks.jsonl
python pipeline/ingest_qdrant.py --input data/processed/sample_chunks.jsonl --batch-size 64
```

O padrão de chunk é **100 tokens com sobreposição de 15 tokens**, medidos pelo tokenizer do modelo de embeddings configurado. Esse valor é conservador em relação ao limite de sequência usual de 128 tokens do modelo selecionado; confirmem o limite real quando alterarem o modelo. O script mantém o texto em lotes de documentos e grava os chunks gradualmente.

### 5. Executar o back-end

Em outro terminal, a partir da raiz do repositório:

```bash
python -m venv .venv-backend
```

Ative o ambiente conforme seu sistema e instale:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Documentação interativa: `http://localhost:8000/docs`.

Para testar somente a interface, defina `MOCK_MODE=true` no `.env`. Os resultados do modo mock são exemplos simulados e **não devem ser usados como medição de desempenho ou avaliação acadêmica**.

### 6. Executar o front-end

Em outro terminal:

```bash
cd frontend
npm install
npm run dev
```

Acesse `http://localhost:5173`.

## Corpus de dados: seleção, download e comprovação dos 10 GB

A fonte definitiva ainda precisa ser escolhida pelo grupo. Antes de baixar, preencham `docs/DATA_SOURCES.md` com a coleção, URL original, licença, data de acesso, formato, filtros e estratégia de armazenamento. Não basta informar o tamanho do arquivo compactado: o requisito é de **pelo menos 10 GB de texto puro**.

Para baixar um arquivo por URL HTTP(S) direta:

```bash
python pipeline/download.py \
  --url "https://ENDERECO-DIRETO/arquivo.jsonl.gz" \
  --filename "corpus.jsonl.gz" \
  --source-name "Nome da coleção" \
  --license "Licença verificada"
```

No PowerShell, o comando pode ser escrito em uma única linha. O download gera `data/raw/download_manifest.json` com URL, tamanho e SHA-256. O script não escolhe a base, não contorna páginas de autenticação e não verifica a licença por conta própria. Para fontes sem URL direta (por exemplo, datasets distribuídos por ferramentas próprias), sigam o método oficial da fonte e registrem o procedimento.

Para medir o conteúdo textual de um JSONL, sem contar as chaves e os metadados do JSON:

```bash
python pipeline/report_corpus.py data/raw/corpus.jsonl.gz
```

Para arquivos de texto puro:

```bash
python pipeline/report_corpus.py --mode text data/raw/parte-1.txt data/raw/parte-2.txt
```

O relatório conta os bytes UTF-8 do campo `text` nos arquivos JSONL. Arquivos `.gz` são lidos sem necessidade de descompactá-los previamente. Conservem o relatório junto aos resultados do projeto para demonstrar o volume efetivo de texto.

## Pré-processamento e indexação do corpus

Pré-processamento em lotes:

```bash
python pipeline/preprocess.py \
  --input data/raw/corpus.jsonl.gz \
  --output data/processed/chunks.jsonl \
  --max-tokens 100 \
  --overlap-tokens 15 \
  --batch-docs 2000
```

Ingestão incremental no Qdrant:

```bash
python pipeline/ingest_qdrant.py \
  --input data/processed/chunks.jsonl \
  --batch-size 64 \
  --encoder-batch-size 32
```

Para recriar a coleção intencionalmente, acrescente `--recreate`. **Atenção:** essa opção apaga todos os pontos da coleção configurada antes de iniciar a carga. Para um teste reduzido, use `--max-records 1000`.

A ingestão gera os embeddings em lotes e envia os pontos ao Qdrant progressivamente. IDs determinísticos tornam a reexecução idempotente para os mesmos pares `doc_id`/`chunk_id`. A coleção e os metadados permanecem no volume local `qdrant_storage/`, ignorado pelo Git.

## Benchmark dos algoritmos

Instale as dependências em ambiente separado ou no ambiente de pipeline:

```bash
pip install -r benchmarks/requirements.txt
```

Execute com o limite padrão de 10.000 documentos:

```bash
python benchmarks/benchmark_search.py \
  --input data/processed/chunks.jsonl \
  --queries-file evaluation/test_questions.json \
  --max-docs 10000 \
  --top-k 5 \
  --query-repeats 3
```

O benchmark mede construção do índice, latência média e P95 (com repetição de consultas), estimativa de memória do índice e, para HNSW, `Recall@K` em relação à busca exata com Sentence Embeddings. Os parâmetros HNSW podem ser alterados por `--hnsw-m`, `--ef-construction` e `--ef-search`.

**Limitação importante:** o benchmark comparativo carrega a amostra selecionada em memória. O padrão de 10.000 documentos é para execução controlada; os resultados devem informar o número de documentos e a máquina usada. `--max-docs 0` tenta carregar tudo e só deve ser usado se houver RAM suficiente. Para estudar o corpus completo, façam experimentos por faixas de tamanho (por exemplo, 1k, 5k, 10k, 25k e 50k) e registrem a memória disponível. A ingestão para o Qdrant, ao contrário, foi projetada para percorrer o JSONL em lotes.

A complexidade descrita no CSV é uma análise teórica simplificada e deve ser explicada no relatório. Para HNSW, não apresentem `O(log N)` como garantia: a busca é aproximada e seu comportamento observado depende da distribuição dos vetores e de parâmetros como `M` e `efSearch`; o pior caso pode se aproximar de uma varredura linear.

Resultados são salvos em `benchmarks/results/metrics.csv`, ignorado pelo Git. Versionem tabelas e gráficos finais escolhidos para o relatório em `docs/`.

## Avaliação com RAGAS

Com a API, Qdrant e Ollama em execução, instale:

```bash
pip install -r evaluation/requirements.txt
python evaluation/eval_ragas.py
```

O script coleta respostas da API e calcula:

- **Faithfulness:** consistência da resposta com o contexto recuperado.
- **Answer relevancy:** relevância da resposta em relação à pergunta.
- **Context precision:** qualidade/ordenação dos contextos recuperados.
- **Context recall:** cobertura das informações necessárias pela recuperação.

Também exporta tempos de resposta com média, mediana (P50) e P95. Os resultados são salvos em `evaluation/results/`.

O arquivo `evaluation/test_questions.json` contém apenas cinco perguntas iniciais: serve para testar o fluxo, não para uma conclusão estatística. Para o experimento final, ampliem para pelo menos 30–50 perguntas variadas, com respostas de referência revisadas pelo grupo. Quando possível, usem um modelo avaliador diferente do modelo de geração (`RAGAS_LLM_MODEL`) e revisem manualmente uma amostra; métricas automáticas dependem da qualidade do avaliador e do conjunto de referência.

## Citações e procedência

Cada chunk mantém `doc_id`, `chunk_id`, título, origem, URL da fonte e texto. A API devolve título, origem, URL, identificador do documento, trecho recuperado e score. O front-end apresenta um link para a fonte original quando a URL estiver disponível.

A LLM recebe os trechos e é instruída a apontar `[Fonte 1]`, `[Fonte 2]` etc. A presença de uma citação gerada no texto não prova, por si só, que a afirmação está correta: confiram se a fonte indicada realmente sustenta a afirmação.

## Testes e verificações

Teste do pré-processamento (ambiente de pipeline com pytest):

```bash
pip install pytest
python -m pytest -q pipeline/tests
```

Verificação manual da API:

- `GET http://localhost:8000/api/health`
- `POST http://localhost:8000/api/ask`

Exemplo de corpo:

```json
{
  "question": "Quando ocorreu a Independência do Brasil?",
  "top_k": 4
}
```

## Organização do trabalho em equipe

Sugestão de branches:

- `feat/dataset-pipeline` — download, medição do corpus e pré-processamento.
- `feat/ingestion-qdrant` — embeddings e ingestão em lotes.
- `feat/benchmarks` — métodos, experimentos e gráficos.
- `feat/backend-frontend` — API, interface, LLM e citações.
- `feat/evaluation` — conjunto de perguntas, RAGAS e análise dos resultados.

Abram Pull Requests para `main`, descrevam como testar e evitem que duas pessoas alterem simultaneamente o mesmo arquivo sem combinar.

## Licenças

Registrem as licenças e versões das bibliotecas, modelos, corpora e componentes em `docs/LICENSES.md`. A licença do código do projeto não substitui as licenças dos dados e dos modelos.

## Preparar o ZIP para o Aprender3 (limite de 100 MB)

O Git não deve conter os 10 GB, os chunks completos, os índices vetoriais, modelos ou ambientes virtuais. Incluam uma amostra pequena e instruções para baixar o corpus completo.

Após revisar o que está versionado:

```bash
git archive --format=zip --output=PAA_PrimeiroNome_Matricula_Proj.zip HEAD
```

Confiram o tamanho do ZIP e incluam o link externo para a base completa e arquivos complementares na entrega.
