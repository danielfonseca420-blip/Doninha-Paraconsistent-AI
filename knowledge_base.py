"""
Base de conhecimento escalável.
===============================
Carrega KB a partir de arquivo (JSON) e opcionalmente enriquece com
retrieval em ChromaDB (RAG). Mantém interface termo -> grau [0,1] para L3/L4.
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# KB padrão (fallback quando não há arquivo)
SEED_KNOWLEDGE_BASE: Dict[str, float] = {
    "quente": 0.85, "frio": 0.85, "morno": 0.70, "aquecido": 0.80, "gelado": 0.80,
    "temperatura": 0.90, "graus": 0.88, "escaldante": 0.75, "tépido": 0.65,
    "verdadeiro": 0.95, "falso": 0.95, "contradição": 0.80, "proposição": 0.85,
    "silogismo": 0.75, "conhecimento": 0.90, "inteligência": 0.85, "consciência": 0.70,
    "razão": 0.88, "verdade": 0.92, "água": 0.95, "líquido": 0.90, "h2o": 0.90,
}


def load_kb_from_file(path: str | Path) -> Dict[str, float]:
    """
    Carrega dicionário termo -> grau [0,1] de um arquivo JSON.
    Formato esperado: {"termo1": 0.9, "termo2": 0.8, ...}
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def merge_kb(base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    """Mescla extra em base; em conflito, extra prevalece."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = v
    return out


def enrich_kb_from_chroma(
    query: str,
    chroma_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 5,
    score_weight: float = 0.8,
) -> Dict[str, float]:
    """
    Busca em ChromaDB por query e retorna um dicionário termo -> peso
    extraído dos trechos recuperados (palavras relevantes com score_weight).
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        return {}

    chroma_path = Path(chroma_path)
    if not chroma_path.exists() or not chroma_path.is_dir():
        return {}

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = Chroma(persist_directory=str(chroma_path), embedding_function=embeddings)
        docs = vectorstore.similarity_search(query, k=k)
    except Exception:
        return {}

    # Extrai termos dos textos e atribui peso
    term_scores: Dict[str, float] = {}
    for d in docs:
        text = d.page_content if hasattr(d, "page_content") else str(d)
        words = re.findall(r"[a-záàãâéêíóôõúüç]+", text.lower())
        for w in words:
            if len(w) > 2:
                term_scores[w] = term_scores.get(w, 0) + score_weight
    # Normaliza para [0, 1]
    if term_scores:
        m = max(term_scores.values())
        term_scores = {t: min(1.0, s / m) for t, s in term_scores.items()}
    return term_scores


def get_knowledge_base(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    query_for_rag: Optional[str] = None,
) -> Dict[str, float]:
    """
    Retorna o KB a ser usado no pipeline.
    - Se config tem knowledge_base.path, carrega desse arquivo.
    - Se config tem chroma_path (ou agent.vector_db_path) e query_for_rag,
      enriquece com retrieval.
    - Fallback: SEED_KNOWLEDGE_BASE.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent
    try:
        from config_loader import load_config, PROJECT_ROOT as _root
        PROJECT_ROOT = _root
        if config is None:
            config = load_config(config_path)
    except Exception:
        pass
    if config is None:
        config = {}

    kb_path = config.get("knowledge_base", {}).get("path") or config.get("knowledge_base", {}).get("path", "")
    chroma_path = config.get("knowledge_base", {}).get("chroma_path") or config.get("agent", {}).get("vector_db_path", "")

    if kb_path and os.path.isabs(kb_path):
        base = load_kb_from_file(kb_path)
    elif kb_path:
        base = load_kb_from_file(PROJECT_ROOT / kb_path)
    else:
        base = dict(SEED_KNOWLEDGE_BASE)

    if not base:
        base = dict(SEED_KNOWLEDGE_BASE)

    if chroma_path and query_for_rag:
        extra = enrich_kb_from_chroma(
            query_for_rag,
            chroma_path,
            config.get("agent", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            k=5,
        )
        base = merge_kb(base, extra)

    return base
