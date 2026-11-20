# Avaliação RAGAS

`test_questions.json` é um conjunto inicial de cinco perguntas para validar a execução do fluxo. Não é suficiente, isoladamente, para sustentar uma conclusão ampla sobre a qualidade do sistema.

Para a avaliação da disciplina:

1. Ampliem o conjunto para pelo menos 30–50 perguntas que cubram períodos, temas e tipos de pergunta variados.
2. Revisem as respostas de referência (`ground_truth`) com fontes confiáveis e guardem a procedência.
3. Coletem resposta, contextos recuperados e latência pela API.
4. Analisem `faithfulness`, `answer_relevancy`, `context_precision` e `context_recall`.
5. Revisem manualmente uma amostra de respostas, especialmente quando as métricas forem baixas ou divergirem da avaliação humana.
6. Registrem qual modelo foi usado para gerar a resposta e qual modelo foi usado como avaliador.

O avaliador pode influenciar os resultados. Sempre que os recursos permitirem, usem um modelo avaliador diferente da LLM que gera as respostas. Não tratem a pontuação automática como verdade absoluta.
