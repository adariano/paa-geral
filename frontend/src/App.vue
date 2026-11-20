<script setup lang="ts">
import { ref } from 'vue';
import { askQuestion } from './services/api';
import type { AnswerResponse } from './types/api';

const question = ref('');
const topK = ref(4);
const loading = ref(false);
const error = ref<string | null>(null);
const result = ref<AnswerResponse | null>(null);

const exampleQuestions = [
  'O que foi a Inconfidência Mineira?',
  'Quando e como ocorreu a Proclamação da República no Brasil?',
  'Qual foi o papel da Lei Áurea na abolição da escravidão?',
];

async function handleSubmit() {
  const trimmed = question.value.trim();
  if (trimmed.length < 3 || loading.value) return;

  loading.value = true;
  error.value = null;

  try {
    result.value = await askQuestion(trimmed, topK.value);
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Erro desconhecido ao consultar o servidor.';
  } finally {
    loading.value = false;
  }
}

function selectExample(example: string) {
  question.value = example;
  handleSubmit();
}
</script>

<template>
  <div class="min-h-screen flex flex-col">
    <header class="bg-emerald-900 text-white shadow-sm">
      <div class="max-w-4xl mx-auto px-4 py-6">
        <p class="text-xs uppercase tracking-wider text-emerald-200 font-semibold">
          UnB · Projeto e Análise de Algoritmos (2/2026)
        </p>
        <h1 class="text-2xl font-bold mt-1">
          Busca Semântica em História Brasileira
        </h1>
        <p class="text-sm text-emerald-100 mt-1">
          Faça perguntas em linguagem natural e receba respostas formatadas por LLM local com citação das fontes.
        </p>
      </div>
    </header>

    <main class="max-w-4xl w-full mx-auto px-4 py-8 flex-1 space-y-6">
      <!-- Formulário de pergunta -->
      <form @submit.prevent="handleSubmit" class="bg-white rounded-xl shadow-sm border border-slate-200 p-5 space-y-4">
        <label for="question" class="block text-sm font-medium text-slate-700">
          Sua pergunta
        </label>
        <div class="flex flex-col sm:flex-row gap-3">
          <input
            id="question"
            v-model="question"
            type="text"
            placeholder="Ex.: Quais foram as principais causas da Independência do Brasil?"
            class="flex-1 rounded-lg border border-slate-300 px-4 py-2.5 text-sm focus:border-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-200"
            :disabled="loading"
          />
          <button
            type="submit"
            :disabled="loading || question.trim().length < 3"
            class="rounded-lg bg-emerald-700 px-5 py-2.5 text-sm font-semibold text-white hover:bg-emerald-800 disabled:opacity-50 transition"
          >
            {{ loading ? 'Consultando...' : 'Perguntar' }}
          </button>
        </div>

        <div class="flex flex-wrap items-center justify-between gap-2 pt-1">
          <div class="flex flex-wrap gap-2">
            <button
              v-for="ex in exampleQuestions"
              :key="ex"
              type="button"
              @click="selectExample(ex)"
              class="text-xs bg-slate-100 hover:bg-slate-200 text-slate-700 px-2.5 py-1 rounded-full transition"
            >
              {{ ex }}
            </button>
          </div>
          <label class="text-xs text-slate-600 flex items-center gap-1.5">
            Fontes (top-k):
            <select v-model.number="topK" class="rounded border border-slate-300 px-1.5 py-0.5 text-xs">
              <option :value="3">3</option>
              <option :value="4">4</option>
              <option :value="5">5</option>
              <option :value="8">8</option>
            </select>
          </label>
        </div>
      </form>

      <!-- Erro -->
      <div v-if="error" class="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        {{ error }}
      </div>

      <!-- Resultado -->
      <section v-if="result" class="space-y-5">
        <article class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
          <div class="flex flex-wrap items-center justify-between gap-2 mb-3">
            <h2 class="text-base font-semibold text-slate-800">Resposta</h2>
            <span class="text-xs font-mono bg-emerald-50 text-emerald-800 border border-emerald-200 px-2.5 py-0.5 rounded-full">
              Modelo: {{ result.model }}
            </span>
          </div>
          <p class="text-slate-700 leading-relaxed whitespace-pre-line">
            {{ result.answer }}
          </p>

          <!-- Métricas de tempo -->
          <div class="mt-5 pt-4 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div class="bg-slate-50 rounded-lg p-2.5">
              <p class="text-slate-500">Embedding</p>
              <p class="font-semibold text-slate-800">{{ result.timings.embedding_ms }} ms</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2.5">
              <p class="text-slate-500">Busca Vetorial</p>
              <p class="font-semibold text-slate-800">{{ result.timings.search_ms }} ms</p>
            </div>
            <div class="bg-slate-50 rounded-lg p-2.5">
              <p class="text-slate-500">Inferência LLM</p>
              <p class="font-semibold text-slate-800">{{ result.timings.llm_ms }} ms</p>
            </div>
            <div class="bg-emerald-50 rounded-lg p-2.5">
              <p class="text-emerald-700">Tempo Total</p>
              <p class="font-semibold text-emerald-900">{{ result.timings.total_ms }} ms</p>
            </div>
          </div>
        </article>

        <!-- Citações -->
        <div class="space-y-3">
          <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-500">
            Documentos Citados ({{ result.citations.length }})
          </h3>
          <div
            v-for="(citation, idx) in result.citations"
            :key="idx"
            class="bg-white rounded-xl border border-slate-200 p-4 shadow-sm"
          >
            <div class="flex items-center justify-between gap-2">
              <h4 class="font-semibold text-sm text-emerald-900">
                [Fonte {{ idx + 1 }}] {{ citation.document }}
              </h4>
              <span class="text-xs font-mono text-slate-500">
                Score: {{ citation.score.toFixed(4) }}
              </span>
            </div>
            <p class="text-xs text-slate-500 mt-0.5">{{ citation.source }}</p>
            <a
              v-if="citation.source_url"
              :href="citation.source_url"
              target="_blank"
              rel="noopener noreferrer"
              class="inline-block mt-1 text-xs text-emerald-700 hover:underline break-all"
            >Consultar documento original</a>
            <p class="text-sm text-slate-700 mt-2 leading-relaxed">
              "{{ citation.snippet }}"
            </p>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>
