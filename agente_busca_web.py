"""
Agente de pesquisa com busca local (ChromaDB) e busca web (DuckDuckGo).
=======================================================================
Utiliza LLM Groq (ReAct), base vetorial local e DuckDuckGo para respostas
precisas, priorizando a base local e complementando com a internet quando necessário.

Requisitos de ambiente:
  - .env com GROQ_API_KEY
  - Base ChromaDB em meu_vector_db (ou configurado em VECTOR_DB_PATH)
  - Dependências: langchain-groq, langchain-chroma, langchain-community,
                  duckduckgo-search, python-dotenv, sentence-transformers
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Carregamento de variáveis de ambiente
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env opcional se as chaves já estiverem no ambiente

# Verificação das chaves obrigatórias antes de importar libs pesadas
def _check_env() -> None:
    """Garante que GROQ_API_KEY está definida."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY não encontrada. Defina no .env ou no ambiente."
        )

_check_env()

# -----------------------------------------------------------------------------
# Imports das bibliotecas do agente
# -----------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.runnables import RunnableConfig
except ImportError as e:
    print("Erro: instale langchain-groq e langchain-core.", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    Chroma = None
    HuggingFaceEmbeddings = None

try:
    from langchain.tools.retriever import create_retriever_tool
except ImportError:
    try:
        from langchain_core.tools import create_retriever_tool
    except ImportError:
        create_retriever_tool = None

try:
    from langchain_community.tools.duckduckgo_search import DuckDuckGoSearchRun
except ImportError:
    DuckDuckGoSearchRun = None

# AgentExecutor e create_react_agent: podem estar em langchain ou langgraph
try:
    from langgraph.prebuilt import create_react_agent as create_react_agent_graph
    USE_LANGGRAPH = True
except ImportError:
    USE_LANGGRAPH = False
    try:
        from langchain.agents import create_react_agent, AgentExecutor
    except ImportError:
        create_react_agent = None
        AgentExecutor = None


# -----------------------------------------------------------------------------
# Configurações
# -----------------------------------------------------------------------------
# Pasta da base ChromaDB (mesmo nome usado em vector_db.py, se existir)
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "meu_vector_db")
# Modelo Groq: "mixtral-8x7b-32768" (mais rápido) ou "llama-3.3-70b-versatile" (mais capaz)
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------
def get_llm() -> ChatGroq:
    """Instancia o ChatGroq com o modelo configurado."""
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )


# -----------------------------------------------------------------------------
# Base vetorial ChromaDB e ferramenta de busca local
# -----------------------------------------------------------------------------
def get_retriever_tool():
    """
    Cria a ferramenta de busca local (ChromaDB).
    Retorna None se a base não existir ou se as dependências não estiverem instaladas.
    """
    if Chroma is None or HuggingFaceEmbeddings is None:
        return None
    if create_retriever_tool is None:
        return None

    persist_dir = Path(VECTOR_DB_PATH)
    if not persist_dir.exists() or not persist_dir.is_dir():
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        return create_retriever_tool(
            retriever,
            name="busca_local",
            description=(
                "Busca informações na base de dados local de treinamento. "
                "Use sempre primeiro antes de buscar na internet."
            ),
        )
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Ferramenta de busca na internet (DuckDuckGo — sem API key)
# -----------------------------------------------------------------------------
def get_duckduckgo_tool():
    """Cria a ferramenta DuckDuckGo para busca na web. Não requer API key."""
    if DuckDuckGoSearchRun is None:
        return None
    try:
        tool = DuckDuckGoSearchRun(
            name="busca_internet",
            description=(
                "Busca informações atualizadas na internet quando a base local "
                "não tem a resposta ou a confiança é baixa. Use para temas atuais e "
                "consulta a páginas web. Não requer chave de API."
            ),
        )
        return tool
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Prompt do sistema (português)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é um agente de pesquisa inteligente.

Regras:
1. Sempre busque primeiro na base local (ferramenta busca_local).
2. Se não encontrar informação suficiente ou a confiança for baixa, use a busca na internet (busca_internet).
3. Responda em português, cite fontes e seja preciso.
4. Se precisar de mais informações, use as ferramentas quantas vezes for necessário.
5. Ao final, apresente uma resposta clara e bem fundamentada."""


# -----------------------------------------------------------------------------
# Construção do agente ReAct
# -----------------------------------------------------------------------------
def build_agent():
    """
    Monta o agente ReAct com as ferramentas disponíveis.
    Usa LangGraph se disponível; caso contrário, AgentExecutor clássico.
    """
    llm = get_llm()
    tools = []

    # Ferramenta 1: busca local
    local_tool = get_retriever_tool()
    if local_tool:
        tools.append(local_tool)
    else:
        print(
            "[AVISO] Base ChromaDB não encontrada ou indisponível. "
            "Apenas busca na internet será usada.",
            file=sys.stderr,
        )

    # Ferramenta 2: busca internet (DuckDuckGo)
    web_tool = get_duckduckgo_tool()
    if web_tool:
        tools.append(web_tool)
    else:
        print(
            "[AVISO] DuckDuckGo não disponível (instale duckduckgo-search). Apenas busca local será usada.",
            file=sys.stderr,
        )

    if not tools:
        raise RuntimeError(
            "Nenhuma ferramenta disponível. Configure ChromaDB (meu_vector_db) ou instale duckduckgo-search."
        )

    if USE_LANGGRAPH:
        # LangGraph: create_react_agent retorna um compilado invocável
        agent = create_react_agent_graph(llm, tools)
        return agent, None, tools

    # LangChain clássico: create_react_agent + AgentExecutor
    if create_react_agent is None or AgentExecutor is None:
        raise RuntimeError(
            "Para usar sem LangGraph, instale langchain com suporte a agents: "
            "pip install langchain langchain-community"
        )

    # Prompt no formato ReAct: input + agent_scratchpad; tools/tool_names são preenchidos pelo executor
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + "\n\nUse as ferramentas quando necessário.\n{tools}\n\nNomes das ferramentas: {tool_names}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )
    return None, executor, tools


# -----------------------------------------------------------------------------
# Invocação unificada (LangGraph ou AgentExecutor)
# -----------------------------------------------------------------------------
def run_agent(query: str, agent_obj, executor, tools):
    """
    Executa o agente com a pergunta do usuário.
    Retorna (resposta_final, passos_intermediarios, fontes).
    """
    if USE_LANGGRAPH and agent_obj is not None:
        from langchain_core.messages import HumanMessage
        config = RunnableConfig(recursion_limit=20)
        result = agent_obj.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        messages = result.get("messages", [])
        # Última mensagem do assistente é a resposta final
        answer = ""
        steps = []
        sources = []
        for m in messages:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("args", {})
                    steps.append({"tool": name, "args": args})
            if hasattr(m, "content") and m.content:
                answer = m.content
            if hasattr(m, "additional_kwargs") and m.additional_kwargs:
                # Tool results podem estar em tool_calls/results
                pass
        if not answer and messages:
            answer = str(messages[-1])
        return answer, steps, sources

    # AgentExecutor (LangChain clássico)
    out = executor.invoke({"input": query})
    answer = out.get("output", "")
    steps = out.get("intermediate_steps", [])
    sources = []
    for step in steps:
        if len(step) >= 2:
            action, observation = step[0], step[1]
            tool_name = getattr(action, "tool", str(action))
            sources.append({"ferramenta": tool_name, "observação": str(observation)[:500]})
    return answer, steps, sources


# -----------------------------------------------------------------------------
# Função para uso pelo pipeline (unificação)
# -----------------------------------------------------------------------------
def run_search_for_context(query: str) -> str:
    """
    Executa o agente de pesquisa e retorna resposta + trechos como um único texto.
    Usado pelo pipeline quando config.agent.use_agent é True.
    """
    try:
        agent_obj, executor, tools = build_agent()
        answer, steps, sources = run_agent(query, agent_obj, executor, tools)
        parts = [answer or ""]
        for s in sources:
            if isinstance(s, dict) and s.get("observação"):
                parts.append(s["observação"][:400])
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Main: interação com o usuário
# -----------------------------------------------------------------------------
def main() -> None:
    """Ponto de entrada: pergunta ao usuário, executa o agente e exibe o resultado."""
    print("=" * 60)
    print("  Agente de Pesquisa — Busca Local + Internet")
    print("=" * 60)

    try:
        agent_obj, executor, tools = build_agent()
    except Exception as e:
        print(f"Erro ao construir o agente: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFerramentas carregadas: {[t.name for t in tools]}")

    # Pergunta ao usuário
    pergunta = input("\nDigite sua pergunta (ou Enter para sair): ").strip()
    if not pergunta:
        print("Nenhuma pergunta informada. Encerrando.")
        return

    print("\n--- Executando agente ---\n")
    try:
        resposta, passos, fontes = run_agent(pergunta, agent_obj, executor, tools)
    except Exception as e:
        print(f"Erro durante a execução: {e}", file=sys.stderr)
        sys.exit(1)

    # Resposta final
    print("\n" + "=" * 60)
    print("  RESPOSTA FINAL")
    print("=" * 60)
    print(resposta)

    # Fontes / observações das ferramentas
    if fontes:
        print("\n" + "-" * 60)
        print("  Fontes / Observações")
        print("-" * 60)
        for i, f in enumerate(fontes, 1):
            if isinstance(f, dict):
                print(f"  [{i}] {f.get('ferramenta', '')}: {f.get('observação', f)[:300]}...")
            else:
                print(f"  [{i}] {str(f)[:300]}")

    # Passos intermediários (resumo)
    if passos:
        print("\n" + "-" * 60)
        print("  Passos intermediários (ReAct)")
        print("-" * 60)
        for i, step in enumerate(passos, 1):
            if isinstance(step, dict):
                print(f"  {i}. Ferramenta: {step.get('tool', '?')}")
                print(f"     Argumentos: {step.get('args', step)}")
            elif isinstance(step, (list, tuple)) and len(step) >= 1:
                action = step[0]
                tool = getattr(action, "tool", "?")
                inp = getattr(action, "tool_input", "") or getattr(action, "input", "")
                print(f"  {i}. Ação: {tool}")
                print(f"     Entrada: {str(inp)[:200]}")
            else:
                print(f"  {i}. {step}")

    print("\n" + "=" * 60)


# -----------------------------------------------------------------------------
# Exemplo de execução
# -----------------------------------------------------------------------------
# No terminal, com .env configurado (GROQ_API_KEY):
#
#   python agente_busca_web.py
#
# O script pede uma pergunta, executa o agente ReAct (busca local primeiro,
# depois internet se necessário) e exibe a resposta final, fontes e passos.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
