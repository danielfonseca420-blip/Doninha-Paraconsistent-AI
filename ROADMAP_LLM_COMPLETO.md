# Roadmap do modelo de LLM completo

**Status:** As sete pendências abaixo foram implementadas (config, KB, L5, unificação com agente, avaliação, chat, API).

---

# O que falta para um modelo de LLM completo (itens implementados)

Este documento lista o que já existe no projeto e o que ainda falta programar para se ter um **modelo de LLM completo** (inferência + geração de texto + RAG + avaliação + entrega).

---

## O que já existe

| Componente | Arquivo(s) | Estado |
|------------|------------|--------|
| Pipeline epistemológico L1→L4 | `pipeline.py`, `l1_concept_table.py`, `l2_kantian_judgments.py`, `syllogism_module.py`, `l3_paraconsistent.py`, `l4_synthesis.py` | ✅ Completo (raciocínio simbólico) |
| Regras paraconsistentes (Fuzzy.txt) | `paraconsistent_rules.py`, `train_truth_model.py` | ✅ L3 treinável por regras |
| Base teórica L4 (Russell) | `l4_russell_equivalence.py`, `train_l4_russell.py` | ✅ Síntese por equivalência |
| Modelo de verdade neural (μ, λ) | `neural_truth_model.py` | ✅ Opcional na L3 |
| LM customizado (TransformerEncoder) | `custom_lm_model.py`, `custom_tokenizer.py`, `pretrain_custom_lm.py` | ✅ Existe mas **não está ligado ao pipeline** |
| Agente de pesquisa (Groq + Chroma + DuckDuckGo) | `agente_busca_web.py` | ✅ Funcional, **separado** do pipeline L1–L4 |
| Banco de conhecimento | `SEED_KNOWLEDGE_BASE` em `pipeline.py` | ⚠️ Estático, dicionário fixo |
| Geração de “resposta” na L4 | `l4_synthesis._generate_response` | ⚠️ Só monta texto a partir da melhor hipótese (template), **não gera texto livre** |

---

## O que falta programar

### 1. **Geração de texto integrada ao pipeline (resposta livre)**

**Problema:** Hoje a L4 devolve um texto do tipo:  
`"Com alta confiança (v=0.85): [melhor proposição] [KB: ...]"` — é um template, não geração token a token.

**O que falta:**

- **Opção A (rápida):** Usar um LLM externo (Groq/OpenAI) na etapa 8:
  - Montar um **prompt estruturado** com: conceitos (L1), juízos filtrados (L2/SYL), valores μ/λ e estado (L3), síntese (L4) e a pergunta do usuário.
  - Chamar o LLM com esse prompt e usar a saída como **resposta final** do pipeline.
- **Opção B (end-to-end):** Integrar o `EpistemicLanguageModel` (ou um decoder pequeno) como gerador:
  - O contexto L1–L4 vira “condição” (ex.: embedding ou texto serializado).
  - O decoder gera a resposta autoregressivamente (`generate_text` já existe em `custom_lm_model.py`).
  - Exige definir formato de contexto (ex.: texto plano com juízos + valores) e treino/ajuste para esse cenário.

**Arquivos a criar/alterar:**  
Novo módulo, ex.: `l5_generation.py` (ou extensão em `l4_synthesis.py`) que recebe `SynthesisResult` + prompt e chama LLM ou `generate_text`; `pipeline.py` passa a usar essa camada na etapa 8.

---

### 2. **Unificar pipeline L1–L4 com o agente de pesquisa**

**Problema:** O pipeline híbrido e o `agente_busca_web.py` são independentes. O pipeline não usa RAG nem busca web.

**O que falta:**

- **Caminho 1:** O pipeline usar o agente como “fonte de fatos”:
  - Antes ou durante L3/L4, chamar o agente (busca local Chroma + DuckDuckGo) com a pergunta ou com as proposições filtradas.
  - Inserir os trechos recuperados no KB ou num contexto extra para a síntese/geração.
- **Caminho 2:** O agente usar o pipeline como “motor de raciocínio”:
  - Após a busca, passar os documentos + pergunta pelo pipeline L1–L4 e usar a `SynthesisResult` para montar o prompt final do LLM do agente.

**Arquivos a criar/alterar:**  
Um orquestrador (ex.: `pipeline_rag.py` ou opções em `pipeline.py`) que instancia tanto `HybridLLMPipeline` quanto o agente e define quando chamar cada um e como combinar saídas.

---

### 3. **Base de conhecimento escalável (KB + RAG)**

**Problema:** O KB é um dicionário fixo em código (`SEED_KNOWLEDGE_BASE`). Não há carregamento a partir de dados nem RAG dentro do pipeline.

**O que falta:**

- Carregar o KB a partir de arquivo(s) (JSON/CSV) ou de um banco.
- Opcional: popular uma base vetorial (ex.: Chroma em `meu_vector_db`) com os mesmos conceitos/evidências e, no pipeline, usar **retrieval** (como no agente) para enriquecer L3/L4 com trechos relevantes à pergunta.
- Manter a interface atual (termo → grau de evidência) para não quebrar L3/L4.

**Arquivos a criar/alterar:**  
Módulo `knowledge_base.py` (ou estender `pipeline.py`) para: carregar KB de disco; opcionalmente conectar a Chroma e expor uma função que, dada a pergunta, retorna um dicionário termo→peso ou trechos para contexto.

---

### 4. **Avaliação e métricas**

**Problema:** Não há como medir qualidade das respostas nem comparar versões do modelo.

**O que falta:**

- Dataset de avaliação: pares (pergunta, resposta_referência) ou (pergunta, valor_verdade_esperado).
- Script de avaliação que:
  - Roda o pipeline (e, se existir, o gerador) em cada pergunta.
  - Calcula métricas, por exemplo:
    - Coerência com L3 (ex.: valor de verdade médio, ausência de trivialização).
    - Se houver referência: similaridade semântica (embedding), BLEU/ROUGE, ou métricas de QA.
- Opcional: relatório (JSON/Markdown) com médias e exemplos.

**Arquivos a criar:**  
`eval_pipeline.py`, pasta `data/eval/` com conjuntos de teste e, se quiser, `metrics.py` com funções de métrica reutilizáveis.

---

### 5. **API ou serviço para uso em produção**

**Problema:** Só existe REPL (`python pipeline.py --repl`). Não há endpoint HTTP nem CLI estável para integração.

**O que falta:**

- **API REST (recomendado):** FastAPI (ou Flask) expondo, por exemplo:
  - `POST /process` — recebe `{"prompt": "..."}` e devolve a resposta do pipeline (e, no futuro, do gerador).
  - Opcional: `POST /agent` — repassa para o agente de pesquisa.
- **CLI:** `python -m pipeline --prompt "..."` (ou similar) que imprime só a resposta final, para uso em scripts.
- Configuração por variáveis de ambiente ou arquivo (YAML/JSON) para modelo, paths do KB, Chroma, etc.

**Arquivos a criar:**  
`api.py` (FastAPI) e/ou entrypoint em `pipeline.py` com `argparse`; opcional `config.yaml`.

---

### 6. **Histórico de diálogo (chat multi-turno)**

**Problema:** O pipeline processa um único prompt. Não há memória de conversa.

**O que falta:**

- Manter uma lista de (role, content) para o usuário e o assistente.
- Para cada nova mensagem: montar contexto (ex.: últimas N trocas ou resumo) e passar ao pipeline (e ao gerador, quando existir).
- Definir se L1–L4 usam só a última pergunta ou o contexto resumido (para não estourar tamanho e custo).

**Arquivos a criar/alterar:**  
Classe ou módulo `chat_session.py` (ou dentro do pipeline) que mantém histórico e chama `process`/gerador com contexto; REPL e API passam a usar essa sessão.

---

### 7. **Persistência e configuração**

**Problema:** Caminhos e opções estão espalhados (hardcoded ou env). Não há um “estado” do sistema salvo de forma única.

**O que falta:**

- Arquivo de configuração (YAML/JSON) para:
  - Caminho do KB, Chroma, modelo L3 (truth_scoring_model.pt), base Russell (l4_russell_concepts.json).
  - Flags: usar agente ou não, usar gerador externo ou LM interno, etc.
- Carregar essa config no arranque do pipeline e da API.
- Documentar onde ficam os artefatos treinados (L3, L4, LM customizado) e como recarregá-los.

**Arquivos a criar/alterar:**  
`config.yaml` (ou `.json`) e função `load_config()` usada em `pipeline.py` e `api.py`.

---

## Priorização sugerida

| Ordem | Item | Impacto | Esforço |
|-------|------|--------|--------|
| 1 | Geração de texto integrada (LLM externo no pipeline) | Alto — resposta natural em vez de template | Médio |
| 2 | Base de conhecimento escalável (carregar KB + opcional RAG) | Alto — dados reais | Médio |
| 3 | Unificar pipeline com agente (RAG no fluxo L1–L4) | Alto — um único sistema | Alto |
| 4 | API REST (FastAPI) | Médio — uso em produção | Baixo |
| 5 | Avaliação (dataset + script + métricas) | Médio — evolução guiada | Médio |
| 6 | Histórico de diálogo | Médio — experiência de chat | Médio |
| 7 | Configuração centralizada e persistência | Baixo — organização | Baixo |

---

## Resumo em uma frase

Hoje você tem um **motor de raciocínio epistemológico (L1–L4)** e um **agente de pesquisa** separado; para um **modelo de LLM completo** falta: **integrar um gerador de texto (LLM externo ou LM interno) ao pipeline**, **conectar o pipeline ao agente e ao KB/RAG**, **avaliar respostas** e **expor tudo via API e configuração clara**.
