"""
CAMADA L3 — Avaliação Paraconsistente
======================================
Implementa a Lógica Anotada de Evidências (LAE / PAL2v) de da Costa & Abe.

Cada proposição recebe um par de anotações:
    μ  ∈ [0,1]  — grau de evidência FAVORÁVEL
    λ  ∈ [0,1]  — grau de evidência CONTRÁRIA

Estados resultantes:
  ┌──────────────────────────────────────────────────┐
  │  Verdadeiro     : μ alto, λ baixo                │
  │  Falso          : μ baixo, λ alto                │
  │  Inconsistente  : μ alto, λ alto  (contradição)  │
  │  Indeterminado  : μ baixo, λ baixo               │
  │  Intermediário  : valores médios  (morno, etc.)  │
  └──────────────────────────────────────────────────┘

Princípio central:
    "Contradição Local + Consistência Global ≠ Trivialização"

A explosão é GENTIL: uma contradição local (quente e frio) não
trivializa o sistema — produz o estado "Intermediário" (morno).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import torch

try:
    from paraconsistent_rules import (
        state_from_rules,
        state_12_to_simple,
        load_rules_from_fuzzy_file,
        ParaconsistentRules,
    )
except Exception:
    state_from_rules = None  # type: ignore
    state_12_to_simple = None  # type: ignore
    load_rules_from_fuzzy_file = None  # type: ignore
    ParaconsistentRules = None  # type: ignore

try:
    # Import opcional do modelo neural; o sistema continua funcional sem ele.
    from neural_truth_model import TruthScoringModel, load_tokenizer, neural_annotations
except Exception:  # pragma: no cover - fallback em ambientes sem transformers
    TruthScoringModel = None  # type: ignore
    load_tokenizer = None  # type: ignore
    neural_annotations = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de limiar
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD_TRUE         = 0.7   # μ ≥ este valor e λ ≤ (1 - este) → Verdadeiro
THRESHOLD_FALSE        = 0.3   # μ ≤ este e λ ≥ (1 - este) → Falso
THRESHOLD_INCONSISTENT = 0.6   # ambos acima → Inconsistente (contradição local)
THRESHOLD_INDETERMINATE= 0.4   # ambos abaixo → Indeterminado


@dataclass
class ParaconsistentValue:
    """
    Valor-verdade paraconsistente para uma proposição.
    Baseado na Lógica Anotada de Evidências (LAE).
    """
    proposition: str
    mu: float          # evidência favorável  ∈ [0,1]
    lam: float         # evidência contrária  ∈ [0,1]

    # ── Graus derivados ──────────────────────────────────────────────── #
    @property
    def certainty(self) -> float:
        """Grau de certeza: Gc = μ − λ   ∈ [−1, 1]"""
        return self.mu - self.lam

    @property
    def contradiction(self) -> float:
        """Grau de contradição: Gct = μ + λ − 1   ∈ [−1, 1]"""
        return self.mu + self.lam - 1.0

    @property
    def state(self) -> str:
        """Estado lógico qualitativo. Usa regras do Fuzzy.txt se disponíveis."""
        if state_from_rules is not None and state_12_to_simple is not None:
            state_12 = state_from_rules(self.mu, self.lam)
            return state_12_to_simple(state_12)
        # Fallback: limiares fixos
        if self.mu >= THRESHOLD_TRUE and self.lam <= (1 - THRESHOLD_TRUE):
            return "Verdadeiro"
        if self.mu <= THRESHOLD_FALSE and self.lam >= (1 - THRESHOLD_FALSE):
            return "Falso"
        if self.mu >= THRESHOLD_INCONSISTENT and self.lam >= THRESHOLD_INCONSISTENT:
            return "Inconsistente_local"   # explosão GENTIL — não trivializa
        if self.mu <= THRESHOLD_INDETERMINATE and self.lam <= THRESHOLD_INDETERMINATE:
            return "Indeterminado"
        return "Intermediário"             # ex: morno entre quente e frio

    @property
    def state_12(self) -> Optional[str]:
        """Estado lógico de 12 valores (reticulado) conforme Fuzzy.txt, se regras carregadas."""
        if state_from_rules is not None:
            return state_from_rules(self.mu, self.lam)
        return None

    @property
    def truth_value(self) -> float:
        """Valor-verdade escalar normalizado para saída final."""
        return round((self.mu + (1 - self.lam)) / 2.0, 4)

    def __str__(self) -> str:
        return (
            f"  μ={self.mu:.3f}  λ={self.lam:.3f}  "
            f"Gc={self.certainty:+.3f}  Gct={self.contradiction:+.3f}  "
            f"v={self.truth_value:.3f}  [{self.state}]\n"
            f"  \"{self.proposition}\""
        )


# ─────────────────────────────────────────────────────────────────────────────
# Motor paraconsistente
# ─────────────────────────────────────────────────────────────────────────────

class ParaconsistentEngine:
    """
    Avalia as hipóteses kantiana (L2) e atribui valores-verdade
    paraconsistentes a cada uma.

    Pode operar em dois modos:
      - Modo heurístico (padrão): usa apenas o banco de conhecimento.
      - Modo neural: se um TruthScoringModel for fornecido, usa o modelo
        para calcular (μ, λ) compatíveis com a Lógica Anotada.
    """

    def __init__(
        self,
        neural_model: Optional["TruthScoringModel"] = None,
        neural_tokenizer=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.neural_model = neural_model
        self.neural_tokenizer = neural_tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(
        self,
        propositions: List[Tuple[str, float]],  # (texto, peso_de_prioridade_L2)
        knowledge_base: Dict[str, float],        # termo → grau de evidência no BD
    ) -> List[ParaconsistentValue]:
        """
        Para cada proposição:
          μ = evidência favorável extraída do banco de dados
          λ = evidência contrária = 1 − f(compatibilidade)
        """
        results: List[ParaconsistentValue] = []
        for prop_text, l2_priority in propositions:
            mu, lam = self._compute_annotations(prop_text, l2_priority, knowledge_base)
            pv = ParaconsistentValue(proposition=prop_text, mu=mu, lam=lam)
            results.append(pv)

        # Ordena por valor-verdade descendente
        results.sort(key=lambda pv: pv.truth_value, reverse=True)
        return results

    # ------------------------------------------------------------------ #
    # Anotação μ / λ                                                       #
    # ------------------------------------------------------------------ #

    def _compute_annotations(
        self,
        text: str,
        l2_priority: float,
        kb: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Calcula (μ, λ) para uma proposição.

        Se um modelo neural estiver disponível, usa-o para obter anotações
        compatíveis com L3. Caso contrário, volta para a heurística original
        baseada apenas no banco de conhecimento e em contradições locais.
        """
        if self.neural_model is not None and self.neural_tokenizer is not None and neural_annotations is not None:
            mu, lam, _, _ = neural_annotations(self.neural_model.to(self.device), self.neural_tokenizer, text)
            # Pequena modulação pela prioridade de L2 para manter a integração
            mu = min(1.0, mu * (0.5 + 0.5 * l2_priority))
            lam = max(0.0, lam * (1.0 - 0.3 * l2_priority))
            return round(mu, 4), round(lam, 4)

        import re
        tokens = set(re.findall(r"[a-záàãâéêíóôõúüç]+", text.lower()))

        kb_scores = [kb.get(t, 0.0) for t in tokens if kb.get(t, 0.0) > 0]
        mu_kb = sum(kb_scores) / len(kb_scores) if kb_scores else 0.3

        contradiction_detected = self._has_antonym_pair(tokens, kb)
        lam_base = 0.8 if contradiction_detected else (1.0 - mu_kb)

        mu = min(1.0, mu_kb * (0.5 + 0.5 * l2_priority))
        lam = max(0.0, lam_base * (1.0 - 0.3 * l2_priority))

        return round(mu, 4), round(lam, 4)

    ANTONYM_PAIRS = [
        ("quente", "frio"), ("quente", "gelado"),
        ("verdadeiro", "falso"), ("real", "fictício"),
        ("afirmativo", "negativo"), ("possível", "impossível"),
    ]

    def _has_antonym_pair(self, tokens: set, kb: Dict[str, float]) -> bool:
        for a, b in self.ANTONYM_PAIRS:
            if a in tokens and b in tokens:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Consistência global: verifica se sistema trivializou                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_global_consistency(values: List[ParaconsistentValue]) -> bool:
        """
        Retorna True se o sistema é globalmente consistente
        (nenhuma trivialização — todos os estados válidos).
        Uma trivialização ocorre se TODAS as proposições são
        'Inconsistente_local' sem nenhum 'Verdadeiro' ou 'Intermediário'.
        """
        states = {pv.state for pv in values}
        if states == {"Inconsistente_local"}:
            return False   # trivialização global
        return True        # explosão gentil — sistema consistente
