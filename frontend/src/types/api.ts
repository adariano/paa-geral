export interface Citation {
  document: string;
  source: string;
  source_url?: string | null;
  doc_id?: string | null;
  snippet: string;
  score: number;
}

export interface Timings {
  embedding_ms: number;
  search_ms: number;
  llm_ms: number;
  total_ms: number;
}

export interface AnswerResponse {
  answer: string;
  citations: Citation[];
  timings: Timings;
  model: string;
}
