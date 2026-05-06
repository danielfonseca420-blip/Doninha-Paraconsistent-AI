"""
Métricas de avaliação do pipeline.
==================================
Coerência L3, similaridade semântica, BLEU/ROUGE (quando disponível).
"""

from __future__ import annotations
import re
from typing import Dict, List, Optional, Any

# Resultado da L4
try:
    from l4_synthesis import SynthesisResult
except Exception:
    SynthesisResult = None  # type: ignore


def coherence_l3(truth_value: float, state: str, contradiction: float) -> Dict[str, float]:
    """
    Métricas de coerência com a camada L3.
    - truth_value alto e contradição baixa = bom.
    - state "Falso" ou "Indeterminado" com truth_value baixo = esperado coerente.
    """
    # Score de coerência: valor alto é bom quando não há trivialização
    contradiction_penalty = abs(contradiction)  # contradição extrema penaliza
    coherence = max(0.0, 1.0 - contradiction_penalty) * (0.5 + 0.5 * truth_value)
    return {
        "coherence_score": round(coherence, 4),
        "truth_value": truth_value,
        "contradiction_abs": abs(contradiction),
    }


def tokenize_pt(text: str) -> List[str]:
    """Tokenização simples para BLEU/ROUGE: palavras em minúsculo."""
    return re.findall(r"[a-záàãâéêíóôõúüç]+", text.lower())


def bleu_sentence(reference: str, hypothesis: str, max_n: int = 2) -> float:
    """
    BLEU simplificado por frase (n-gram precision até max_n).
    Retorna valor em [0, 1].
    """
    ref_tok = tokenize_pt(reference)
    hyp_tok = tokenize_pt(hypothesis)
    if not hyp_tok:
        return 0.0
    if not ref_tok:
        return 0.0
    p_n = []
    for n in range(1, max_n + 1):
        ref_ngrams = [tuple(ref_tok[i : i + n]) for i in range(len(ref_tok) - n + 1)]
        hyp_ngrams = [tuple(hyp_tok[i : i + n]) for i in range(len(hyp_tok) - n + 1)]
        if not hyp_ngrams:
            continue
        matches = sum(1 for g in hyp_ngrams if g in ref_ngrams)
        p_n.append(matches / len(hyp_ngrams))
    if not p_n:
        return 0.0
    # Média geométrica das precisions
    prod = 1.0
    for p in p_n:
        prod *= p
    return prod ** (1.0 / len(p_n))


def rouge_l_sentence(reference: str, hypothesis: str) -> float:
    """
    ROUGE-L simplificado (LCS de palavras).
    Retorna F1 em [0, 1].
    """
    ref_tok = tokenize_pt(reference)
    hyp_tok = tokenize_pt(hypothesis)
    if not ref_tok or not hyp_tok:
        return 0.0
    # LCS por palavras
    m, n = len(ref_tok), len(hyp_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tok[i - 1] == hyp_tok[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / n if n else 0
    rec = lcs / m if m else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def semantic_similarity(reference: str, hypothesis: str) -> float:
    """
    Similaridade por embeddings (se sentence-transformers disponível).
    Caso contrário, retorna -1.0 para indicar indisponível.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        ref_emb = model.encode(reference)
        hyp_emb = model.encode(hypothesis)
        from numpy import dot
        from numpy.linalg import norm
        return float(dot(ref_emb, hyp_emb) / (norm(ref_emb) * norm(hyp_emb) + 1e-9))
    except Exception:
        return -1.0


def evaluate_response(
    synthesis_result: "SynthesisResult",
    reference_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Agrega métricas: coerência L3 + BLEU/ROUGE (e opcionalmente similaridade)
    quando há resposta de referência.
    """
    out: Dict[str, Any] = {}
    out["coherence"] = coherence_l3(
        synthesis_result.truth_value,
        synthesis_result.state,
        synthesis_result.contradiction,
    )
    if reference_answer:
        out["bleu"] = round(bleu_sentence(reference_answer, synthesis_result.response), 4)
        out["rouge_l"] = round(rouge_l_sentence(reference_answer, synthesis_result.response), 4)
        sim = semantic_similarity(reference_answer, synthesis_result.response)
        if sim >= 0:
            out["semantic_similarity"] = round(sim, 4)
    return out
