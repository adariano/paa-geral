# Registro da fonte de dados

A fonte definitiva da base de História Brasileira ainda precisa ser decidida pelo grupo. Preencham esta ficha antes do download e mantenham uma cópia preenchida nos documentos da apresentação.

| Campo | Informação do grupo |
|---|---|
| Nome da base/coleção | PREENCHER |
| Instituição ou mantenedor | PREENCHER |
| URL oficial da coleção | PREENCHER |
| URL direta dos arquivos | PREENCHER |
| Data de acesso | PREENCHER |
| Licença dos dados | VERIFICAR E PREENCHER |
| Termos de uso/atribuição | PREENCHER |
| Formato dos arquivos | PREENCHER |
| Idioma(s) | PREENCHER |
| Estratégia para selecionar História Brasileira | PREENCHER |
| Volume total baixado (bytes) | PREENCHER |
| Volume de texto puro (bytes UTF-8) | PREENCHER |
| Hashes SHA-256 | Manifesto em `data/raw/download_manifest.json` ou arquivo complementar |
| Link externo para download | PREENCHER |

## Critérios de seleção

- A fonte deve permitir legalmente o uso e a redistribuição pretendidos pelo trabalho.
- A temática deve ser efetivamente História Brasileira; justifiquem os filtros e a cobertura temporal.
- Verifiquem idioma, duplicatas, qualidade do texto, metadados e referências a documentos de origem.
- Documentem se o tamanho informado é compactado, descompactado ou o total de texto do campo textual.
- O requisito acadêmico é pelo menos 10 GB de texto puro. Usem `pipeline/report_corpus.py` para registrar o tamanho do conteúdo textual e não apenas o tamanho do arquivo compactado.

## Formato intermediário recomendado

JSON Lines (um objeto por linha), com pelo menos os campos:

```json
{
  "doc_id": "identificador-estavel-da-fonte",
  "title": "Título do documento",
  "source": "Nome da coleção ou instituição",
  "source_url": "https://...",
  "text": "Conteúdo textual do documento"
}
```

Não removam a URL original nem a informação de licença durante o processamento. A URL é utilizada para apresentar a procedência na interface.
