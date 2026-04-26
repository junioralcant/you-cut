---
name: executar-todas-tasks
description: Executes all pending tasks sequentially from tasks.md, one at a time, using executar-task as the standard implementation skill. After each task, runs task-review and only advances to the next task if the review result is APPROVED or APPROVED WITH OBSERVATIONS. Halts the pipeline on REJECTED reviews and waits for user action. Use when the user asks to run all tasks, execute the full task pipeline, or automate sequential task delivery.
---

# Execução Sequencial de Todas as Tasks

## Objetivo

Executar todas as tasks pendentes de um feature de forma sequencial e controlada, garantindo que cada task passe por implementação (via `executar-task`) e revisão (via `task-review`) antes de avançar para a próxima.

## Procedures

**Step 1: Localizar e Carregar o Contexto do Feature (Obrigatório)**

1. Leia o arquivo `./tasks/prd-[feature-slug]/tasks.md` para obter a lista completa de tasks.
2. Identifique todas as tasks com status pendente (não marcadas como completas).
3. Ordene as tasks pela numeração para garantir execução na sequência correta.
4. Leia o PRD em `./tasks/prd-[feature-slug]/prd.md` e o Tech Spec em `./tasks/prd-[feature-slug]/techspec.md` para ter contexto completo antes de iniciar.
5. Se não houver tasks pendentes, informe ao usuário e encerre.

**Step 2: Exibir Plano de Execução (Obrigatório)**

1. Liste todas as tasks a serem executadas com seus IDs e nomes.
2. Informe ao usuário o total de tasks pendentes.
3. Apresente o fluxo: Task N → Implementação → Review → (se aprovada) Task N+1.

**Step 3: Loop de Execução (Obrigatório — repetir para cada task pendente)**

Para cada task na lista ordenada:

**3a. Anunciar Task Atual**
- Informe ao usuário qual task está sendo iniciada (ID + nome).
- Exiba o progresso: "Task X de Y".

**3b. Executar a Task**
- Siga integralmente o workflow definido na skill `executar-task/SKILL.md`.
- Leia `../executar-task/SKILL.md` e execute todos os passos obrigatórios:
  - Pre-Task Configuration (leitura do task file, PRD, Tech Spec, dependências).
  - Load Required Skills.
  - Task Analysis.
  - Approach Plan.
  - Implementation.
  - Mark Task Complete em `tasks.md`.
- Não avance para a revisão se a implementação falhar ou os testes não passarem.

**3c. Executar Review da Task**
- Após a implementação bem-sucedida, execute o workflow completo da skill `task-review/SKILL.md`.
- Leia `../task-review/SKILL.md` e siga todos os passos.
- O review deve produzir um dos três vereditos: `APPROVED`, `APPROVED WITH OBSERVATIONS` ou `REJECTED`.

**3d. Avaliar Resultado do Review**

- Se `APPROVED` ou `APPROVED WITH OBSERVATIONS`:
  - Registre o resultado positivo.
  - Informe ao usuário que a task foi concluída e aprovada.
  - **Execute `/clear` para limpar todo o contexto da conversa antes de iniciar a próxima task.**
  - Avance para a próxima task (volte ao passo 3a).

- Se `REJECTED`:
  - **Interrompa o pipeline imediatamente.**
  - Apresente ao usuário o relatório de review completo com todos os problemas encontrados.
  - Aguarde instrução do usuário sobre como proceder.
  - **Não execute a próxima task** até que o usuário resolva os problemas e solicite retomada.
  - Após correções do usuário, execute `/clear` para limpar o contexto e repita apenas o passo 3b (re-implementação) e 3c (re-review) para a task rejeitada antes de continuar.

**Step 4: Relatório Final (Obrigatório)**

Ao concluir todas as tasks (ou ao ser interrompido por um REJECTED não resolvido), gere um relatório com:

1. **Resumo Geral**: Total de tasks, quantas foram aprovadas, quantas rejeitadas, status final.
2. **Tabela de Tasks**: Para cada task: ID | Nome | Status de Implementação | Resultado do Review.
3. **Próximos Passos**: Se todas aprovadas, indique que o feature está pronto para QA. Se houver tasks pendentes, liste-as.

## Regras de Controle de Fluxo

- **Sequencialidade estrita**: nunca execute a task N+1 antes de a task N estar aprovada.
- **Limpeza de contexto obrigatória**: execute `/clear` após cada task aprovada, antes de iniciar a próxima. Isso libera o contexto acumulado e garante que cada task começa com contexto limpo e foco total.
- **Bloqueio por REJECTED**: um veredito REJECTED congela o pipeline. O usuário deve intervir.
- **Sem pular tasks**: mesmo que uma task pareça trivial, ela deve passar por implementação e review.
- **Dependências**: se uma task depende de outra ainda não concluída, reporte ao usuário e interrompa.

## Error Handling

- Se o arquivo `tasks.md` não for encontrado, encerre e informe o usuário.
- Se a feature slug não puder ser determinada, solicite ao usuário que informe o caminho correto.
- Se a implementação de uma task falhar (testes não passam, erro crítico), trate como REJECTED implícito: interrompa e reporte ao usuário.
- Se o review não produzir um veredito claro, solicite ao usuário que avalie manualmente antes de continuar.
