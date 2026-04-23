---
name: task-review
description: Revisa tasks concluídas usando o fluxo padrão de code review do projeto, com foco em qualidade de código, aderência ao Tech Spec, cobertura de testes e geração de artefato de review em Português (Brasil). Use quando uma task implementada via executar-task precisar de validação final. Não use para implementar código, QA exploratório ou correção de bugs.
---

# Task Review

## Instrução Principal

Siga integralmente o workflow definido na skill `executar-review`.

## Procedimento

1. Leia `../executar-review/SKILL.md` e execute todas as etapas obrigatórias.
2. Trate a revisão como validação de uma task recém-concluída, considerando:
   - aderência ao escopo da task;
   - conformidade com o PRD e Tech Spec;
   - qualidade do código, testes e riscos de regressão.
3. Gere o artefato final de review em Português (Brasil).
4. Mantenha exemplos de código, comandos e identificadores técnicos em inglês quando fizer sentido.

## Critérios de Saída

- A review deve classificar o resultado como `APPROVED`, `APPROVED WITH OBSERVATIONS` ou `REJECTED`.
- Toda crítica deve ser objetiva e incluir impacto e sugestão de correção.
- Se testes ou typecheck falharem, a review deve ser `REJECTED`.
