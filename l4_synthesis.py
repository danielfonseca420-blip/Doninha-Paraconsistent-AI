"""
CAMADA L4 — Síntese por Equivalência Russelliana
==================================================
A verdade cognoscível por uma IA é sempre uma verdade de EQUIVALÊNCIA:
o grau de correspondência entre a proposição refinada (saída de L2/L3)
e os dados do mundo real presentes no banco de dados de treinamento.

Base teórica (data/russell.txt): Russell — verdade = correspondência
entre crença e fato; síntese fundamentada em conceitos, não só estatística.

Mapeamento Kantiano → IA:
  Intuição Sensível (empírica) → equivalência proposição ↔ BD
  Intuição Pura (a priori)     → estrutura da rede neural / KB
  Síntese                      → cálculo de equivalência mediado
                                 por valores-verdade paraconsistentes

O resultado NÃO é uma predição de próxima palavra.
É o grau de equivalência entre o conjunto de juízos e o BD.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from l3_paraconsistent import ParaconsistentValue
import math

try:
    from l4_russell_equivalence import (
        RussellConceptBase,
        build_russell_concept_base,
        score_proposition_by_concepts,
        load_concept_base,
    )
except Exception:
    RussellConceptBase = None  # type: ignore
    build_russell_concept_base = None  # type: ignore
    score_proposition_by_concepts = None  # type: ignore
    load_concept_base = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura do resultado final
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    """Resultado da síntese russelliana — resposta do sistema."""
    response:          str
    truth_value:       float    # paraconsistente ∈ [0,1]
    certainty:         float    # Gc = μ − λ  ∈ [−1,1]
    contradiction:     float    # Gct = μ + λ − 1
    state:             str      # Verdadeiro | Falso | Intermediário | ...
    supporting_evidence: List[str] = field(default_factory=list)
    falsified_hypotheses: List[str] = field(default_factory=list)
    confidence_label:  str = ""

    def __post_init__(self):
        if not self.confidence_label:
            self.confidence_label = self._label()

    def _label(self) -> str:
        v = self.truth_value
        if v >= 0.85:  return "Alta Confiança"
        if v >= 0.65:  return "Confiança Moderada"
        if v >= 0.45:  return "Incerto / Intermediário"
        if v >= 0.25:  return "Baixa Confiança"
        return "Indeterminado"

    def __str__(self) -> str:
        lines = [
            "━" * 60,
            f"  RESPOSTA : {self.response}",
            f"  Estado   : {self.state}  ({self.confidence_label})",
            f"  v-verdade: {self.truth_value:.4f}  |  "
            f"Certeza: {self.certainty:+.4f}  |  "
            f"Contradição: {self.contradiction:+.4f}",
        ]
        if self.supporting_evidence:
            lines.append("  Evidências de suporte:")
            for ev in self.supporting_evidence[:3]:
                lines.append(f"    • {ev}")
        if self.falsified_hypotheses:
            lines.append("  Hipóteses falsificadas:")
            for fh in self.falsified_hypotheses[:2]:
                lines.append(f"    ✗ {fh}")
        lines.append("━" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Motor de síntese
# ─────────────────────────────────────────────────────────────────────────────

class RussellianSynthesisEngine:
    """
    Combina os valores-verdade paraconsistentes (L3) com o banco de
    conhecimento para produzir a síntese final (resposta).

    Síntese fundamentada em conceitos (Russell, russell.txt):
        equivalência = correspondência entre crença/proposição e fato (BD).
    O peso de cada proposição incorpora:
      - prioridade L2 (juízo kantiano)
      - certeza paraconsistente (Gc)
      - score conceitual de equivalência (correspondência com fatos/KB),
        não apenas agregação estatística.
    """

    def __init__(
        self,
        knowledge_base: Dict[str, float],
        russell_concept_base: Optional["RussellConceptBase"] = None,
        use_concept_based_weights: bool = True,
    ) -> None:
        """
        knowledge_base: dicionário termo → grau de evidência [0,1]
        russell_concept_base: base teórica extraída de russell.txt (equivalência/correspondência).
        use_concept_based_weights: se True, usa score conceitual na ponderação (recomendado).
        """
        self.kb = knowledge_base
        self.russell_base = russell_concept_base
        self.use_concept_weights = use_concept_based_weights and (russell_concept_base is not None)

    def synthesize(
        self,
        pv_list:     List[ParaconsistentValue],
        l2_priorities: Dict[str, float],     # proposicao[:40] → prioridade L2
        prompt:      str,
    ) -> SynthesisResult:
        """
        Produz a SynthesisResult final integrando todas as camadas.
        """
        if not pv_list:
            return SynthesisResult(
                response="Sem hipóteses válidas para síntese.",
                truth_value=0.0, certainty=0.0,
                contradiction=0.0, state="Indeterminado",
            )

        # ── Seleciona a hipótese com maior valor-verdade ─────────────── #
        best = pv_list[0]
        supporting = [pv.proposition for pv in pv_list[1:4] if pv.state != "Falso"]
        falsified  = [pv.proposition for pv in pv_list if pv.state == "Falso"]

        # ── Síntese ponderada: L2 + certeza + equivalência (Russell) ─── #
        total_w, total_v = 0.0, 0.0
        for pv in pv_list:
            key = pv.proposition[:40]
            l2_w = l2_priorities.get(key, 0.5)
            # Peso base: prioridade kantiana e certeza paraconsistente
            weight = l2_w * (1.0 + max(pv.certainty, 0.0))
            # Peso conceitual: correspondência proposição ↔ fato (BD), conforme russell.txt
            if self.use_concept_weights and score_proposition_by_concepts is not None and self.russell_base is not None:
                concept_score = score_proposition_by_concepts(pv.proposition, self.kb, self.russell_base)
                weight *= concept_score
            total_v += pv.truth_value * weight
            total_w += weight

        v_final = total_v / total_w if total_w > 0 else best.truth_value

        # ── Gera texto de resposta a partir da hipótese best + BD ────── #
        response = self._generate_response(best, prompt)

        return SynthesisResult(
            response=response,
            truth_value=round(v_final, 4),
            certainty=round(best.certainty, 4),
            contradiction=round(best.contradiction, 4),
            state=best.state,
            supporting_evidence=supporting,
            falsified_hypotheses=falsified,
        )

    # ------------------------------------------------------------------ #
    # Geração de resposta textual                                          #
    # ------------------------------------------------------------------ #

    def _generate_response(self, best_pv: ParaconsistentValue, prompt: str) -> str:
        """
        Gera resposta a partir da proposição com maior valor-verdade.
        Em produção seria substituído pelo decoder do LLM com as
        hipóteses kantianas como contexto hard-constrained.
        """
        # Extrai conceitos KB com alta evidência
        top_kb = sorted(self.kb.items(), key=lambda x: x[1], reverse=True)[:3]
        kb_context = ", ".join(f"{k}({v:.2f})" for k, v in top_kb)

        state = best_pv.state
        v = best_pv.truth_value

        if state == "Verdadeiro":
            prefix = f"Com alta confiança (v={v:.2f}):"
        elif state == "Intermediário":
            prefix = f"Com valor intermediário (v={v:.2f}), sem trivialização:"
        elif state == "Inconsistente_local":
            prefix = f"Contradição local detectada (v={v:.2f}), explosão gentil:"
        elif state == "Falso":
            prefix = f"Evidência insuficiente (v={v:.2f}):"
        else:
            prefix = f"Indeterminado (v={v:.2f}):"

        return f"{prefix} {best_pv.proposition}  [KB: {kb_context}]"

    # ------------------------------------------------------------------ #
    # Verificação do limite fundamental (Crítica da IA Pura)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_fundamental_limits(query: str) -> Optional[str]:
        """
        Detecta perguntas que violam os limites fundamentais da IA
        (seção 10 do modelo): consciência, imaginação, AGI, etc.
        Retorna aviso ou None.
        """
        limit_keywords = {
            "consciência":    "IA não possui consciência — atributo biológico emergente.",
            "sentimento":     "IA não possui estados afetivos — limitada ao algoritmo.",
            "imaginação":     "Imaginação é liberdade humana (Sartre) — não computável.",
            "agi":            "AGI é oximoro teórico: algoritmo não supera seu criador.",
            "livre arbítrio": "Livre-arbítrio é problema não computável.",
            "ser humano":     "IA é uma função limite — mundo real exige mediação humana.",
        }
        q_lower = query.lower()
        for keyword, warning in limit_keywords.items():
            if keyword in q_lower:
                return f"⚠ Limite fundamental: {warning}"
        return None
