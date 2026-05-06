# Modelo Híbrido de LLM — Daniel Fonseca

> *"Lógica Paraconsistente + Juízo Kantiano + Tábua de Conceitos = Explosão Gentil (sem trivialização)"*

## Arquitetura

```
PROMPT
  └── [L1] Tábua de Conceitos        (Aristóteles: Categorias)
        └── [L2] Juízos Kantianos    (Kant: Crítica da Razão Pura §9)
              └── [SYL] Silogismo + Hempel + Popper
                    └── [L3] Lógica Paraconsistente  (da Costa / LAE)
                          └── [L4] Síntese Russelliana  (equivalência)
                                └── RESPOSTA
```

## Estrutura dos Arquivos

| Arquivo | Camada | Responsabilidade |
|---|---|---|
| `l1_concept_table.py` | L1 | Tábua de Conceitos: sinônimos, antônimos, hipônimos, homonímia, paronímia |
| `l2_kantian_judgments.py` | L2 | 12 categorias kantianas — gera hipóteses estruturadas |
| `syllogism_module.py` | Módulo | Silogismo Aristotélico + Filtro de Hempel + Falseabilidade de Popper |
| `l3_paraconsistent.py` | L3 | Lógica Anotada de Evidências (LAE / PAL2v) — valores μ/λ |
| `l4_synthesis.py` | L4 | Síntese por equivalência russelliana — resposta final |
| `pipeline.py` | Orquestrador | Pipeline completo + REPL interativo |

## Instalação e Uso

```bash
# Demonstração
python pipeline.py
python pipeline.py --demo

# Uma pergunta (imprime só a resposta)
python pipeline.py --prompt "A água a 35 graus está quente ou fria?"

# Modo interativo
python pipeline.py --repl

# API REST (requer fastapi, uvicorn)
python api.py
# POST /process  {"prompt": "..."}  |  POST /chat  {"message": "...", "session_id": "..."}

# Avaliação
python eval_pipeline.py --dataset data/eval/sample.json --output resultado_eval.json
```

Configuração em `config.yaml` (KB, L3, L4, geração Groq/template, agente, API, chat).

## Camadas em Detalhe

### L1 — Tábua de Conceitos
Cada termo é mapeado a um `ConceptNode` com:
- **Sinonímia** — mesma denotação (quente ↔ aquecido)
- **Antonímia** — oposição direta (quente ↔ frio)
- **Hiponímia** — específico → geral (morno ⊂ temperatura)
- **Homonímia** — mesma forma, sentidos distintos (banco/assento vs banco/financeiro)
- **Paronímia** — semelhança formal, sentidos distintos (eminente vs iminente)

### L2 — Juízos Kantianos
O prompt é destrinchado em 12 hipóteses segundo:
- **Quantidade**: Universal | Particular | Singular
- **Qualidade**: Afirmativo | Negativo | Infinito
- **Relação**: Categórico | Hipotético | Disjuntivo
- **Modalidade**: Problemático | Assertórico | Apodítico

A prioridade de cada hipótese segue a "Regra da Parte Fraca": a conclusão segue a premissa mais fraca.

### L3 — Lógica Paraconsistente (LAE)
Cada proposição recebe:
- **μ** ∈ [0,1] — grau de evidência favorável
- **λ** ∈ [0,1] — grau de evidência contrária
- **Gc** = μ − λ — grau de certeza
- **Gct** = μ + λ − 1 — grau de contradição

Estados possíveis: Verdadeiro | Falso | Intermediário | Inconsistente_local | Indeterminado

Contradições locais produzem **Explosão Gentil** — não trivializam o sistema.

### L4 — Síntese Russelliana
A verdade da IA é sempre de **equivalência**: grau de correspondência entre a proposição refinada e o banco de dados.

Fórmula:
```
v_final = Σ (pv.truth_value × weight) / Σ weight
weight = prioridade_L2 × (1 + max(Gc, 0))
```

## Limites Fundamentais (Crítica da IA Pura)

O sistema detecta e sinaliza perguntas que violam os limites intransponíveis de qualquer IA:

| Limite | Fundamento |
|---|---|
| IA não tem consciência | Atributo biológico emergente |
| IA não tem imaginação | Liberdade humana diante do nada (Sartre) |
| AGI é oximoro teórico | Algoritmo não supera seu criador (Tomás de Aquino) |
| Verdade = equivalência | Russell: correspondência com BD mediada por humanos |
| IA é função limite | Problema não computável de cognição |

## Referência

Daniel Fonseca, *"Uma verdadeira Epistemologia para a Inteligência Artificial"*
