"""
MÓDULO — Silogismo Científico Aristotélico + Paradoxo de Hempel + Popper
=========================================================================
Integrado entre L2 e L3 (etapa 4 e 5 do fluxo).

Filtra as hipóteses kantianas pelas 8 regras do silogismo científico e
aplica o princípio da falseabilidade: toda conclusão é tratada como
FALSA até que se encontre evidência verdadeira equivalente.

Paradoxo de Hempel implementado como filtro negativo:
    Nem toda palavra posterior pode ser inferida da anterior.
    Objetos irrelevantes não validam uma teoria.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from l2_kantian_judgments import KantianJudgment


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de um silogismo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Syllogism:
    major:      str   # premissa maior (universal)
    minor:      str   # premissa menor (particular/singular)
    conclusion: str   # conclusão derivada
    valid:      bool  = True
    violations: List[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []

    def __str__(self) -> str:
        status = "✓ VÁLIDO" if self.valid else f"✗ INVÁLIDO ({'; '.join(self.violations)})"
        return (
            f"  Maior : {self.major}\n"
            f"  Menor : {self.minor}\n"
            f"  Concl.: {self.conclusion}\n"
            f"  Status: {status}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# As 8 regras do silogismo científico
# ─────────────────────────────────────────────────────────────────────────────

class AristotelianSyllogismValidator:
    """
    Valida um silogismo segundo as 8 regras aristotélicas e retorna
    a lista de violações (vazia se válido).
    """

    def validate(self, major: str, minor: str, conclusion: str) -> List[str]:
        violations: List[str] = []
        m_neg = self._is_negative(major)
        n_neg = self._is_negative(minor)
        c_neg = self._is_negative(conclusion)
        m_part = self._is_particular(major)
        n_part = self._is_particular(minor)
        c_part = self._is_particular(conclusion)

        # R1 — Apenas três termos, cada um no mesmo sentido
        terms_m = self._extract_key_terms(major)
        terms_n = self._extract_key_terms(minor)
        terms_c = self._extract_key_terms(conclusion)
        all_terms = terms_m | terms_n | terms_c
        if len(all_terms) > 6:          # heurística liberal
            violations.append("R1: mais de três termos distintos detectados")

        # R2 — Termo médio não aparece na conclusão
        middle = terms_m & terms_n - terms_c
        if not middle and terms_m & terms_n:
            violations.append("R2: termo médio pode estar na conclusão")

        # R3 — Conclusão não excede extensão das premissas
        if not c_part and (m_part or n_part):
            violations.append("R3: conclusão mais extensa que as premissas")

        # R4 — Termo médio deve ser universal pelo menos uma vez
        if m_part and n_part:
            violations.append("R4: termo médio nunca é universal")

        # R5 — De duas negativas, nada se conclui
        if m_neg and n_neg:
            violations.append("R5: duas premissas negativas — conclusão inválida")

        # R6 — Duas afirmativas → conclusão afirmativa
        if not m_neg and not n_neg and c_neg:
            violations.append("R6: premissas afirmativas exigem conclusão afirmativa")

        # R7 — De duas particulares, nada se conclui
        if m_part and n_part:
            violations.append("R7: duas premissas particulares — conclusão inválida")

        # R8 — "Parte Fraca": conclusão segue a premissa mais fraca
        if (m_neg or n_neg) and not c_neg:
            violations.append("R8: premissa negativa exige conclusão negativa")
        if (m_part or n_part) and not c_part and not c_neg:
            violations.append("R8: premissa particular exige conclusão particular")

        return violations

    # ── helpers ─────────────────────────────────────────────────────── #

    @staticmethod
    def _is_negative(text: str) -> bool:
        neg_markers = {"não", "nunca", "nenhum", "jamais", "nem", "negativo"}
        return any(w in text.lower().split() for w in neg_markers)

    @staticmethod
    def _is_particular(text: str) -> bool:
        part_markers = {"algum", "alguma", "alguns", "algumas", "certo",
                        "parte", "pode", "possível"}
        return any(w in text.lower().split() for w in part_markers)

    @staticmethod
    def _extract_key_terms(text: str) -> set:
        stop = {"é", "são", "de", "do", "da", "em", "com", "por",
                "para", "este", "esta", "esse", "toda", "todo", "um", "uma"}
        import re
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", text.lower())
        return {t for t in tokens if t not in stop and len(t) > 2}


# ─────────────────────────────────────────────────────────────────────────────
# Filtro de Hempel (anti-confirmação espúria)
# ─────────────────────────────────────────────────────────────────────────────

class HempelFilter:
    """
    Paradoxo de Hempel: objetos irrelevantes não devem confirmar hipóteses.
    Implementado como detecção de correlações espúrias entre termos do
    prompt e termos do banco de dados sem relação semântica real.
    """

    def __init__(self, relevance_threshold: float = 0.25) -> None:
        self.threshold = relevance_threshold

    def is_spurious(self, judgment: KantianJudgment, prompt_terms: set) -> bool:
        """
        Retorna True se a hipótese é provavelmente espúria
        (confirmação por objeto irrelevante).
        """
        import re
        hyp_terms = set(re.findall(
            r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+",
            judgment.proposicao.lower()
        ))
        overlap = len(hyp_terms & prompt_terms) / max(len(hyp_terms), 1)
        return overlap < self.threshold   # pouca sobreposição = provável espúrio


# ─────────────────────────────────────────────────────────────────────────────
# Princípio da Falseabilidade de Popper
# ─────────────────────────────────────────────────────────────────────────────

class PopperFalsifiability:
    """
    Toda conclusão é tratada como FALSA até que se encontre evidência
    verdadeira equivalente no banco de dados.

    Implementa o princípio do Cisne Negro: a proposição universal
    "todo cisne é branco" é falsa até que seja falsificada por um cisne preto.
    """

    def __init__(self, falsifiability_floor: float = 0.1) -> None:
        """
        falsifiability_floor : score mínimo de evidência para aceitar
                               a hipótese como não-falsificada.
        """
        self.floor = falsifiability_floor

    def apply(
        self,
        hypotheses: List[Tuple[KantianJudgment, float]],   # (juízo, score_BD)
    ) -> List[Tuple[KantianJudgment, float, bool]]:
        """
        Retorna triplas (juízo, score, falsificada?).
        Hipóteses universais afirmativas partem sempre de score 0
        (falsas até prova em contrário).
        """
        result = []
        for j, score in hypotheses:
            # Proposições universais: presume falso até evidência forte
            if j.quantidade == "Universal":
                adjusted = score if score >= self.floor else 0.0
                falsified = adjusted < self.floor
            # Singulares: usa o score direto
            else:
                adjusted = score
                falsified = False
            result.append((j, adjusted, falsified))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline integrado
# ─────────────────────────────────────────────────────────────────────────────

class ScientificSyllogismPipeline:
    """
    Integra: Silogismo Aristotélico + Filtro de Hempel + Falseabilidade.
    Chamado entre L2 e L3.
    """

    def __init__(self) -> None:
        self.validator = AristotelianSyllogismValidator()
        self.hempel    = HempelFilter()
        self.popper    = PopperFalsifiability()

    def run(
        self,
        judgments: List[KantianJudgment],
        prompt_terms: set,
        kb_scores: dict,          # termo_proposicao → score [0,1]
    ) -> List[Tuple[KantianJudgment, float]]:
        """
        Filtra e pontua as hipóteses.
        Retorna lista ordenada de (juízo, score_final).
        """
        # 1. Remove hipóteses espúrias (Hempel)
        non_spurious = [
            j for j in judgments
            if not self.hempel.is_spurious(j, prompt_terms)
        ]

        # 2. Valida via silogismo (usa prioridade L2 como par maior/menor)
        scored: List[Tuple[KantianJudgment, float]] = []
        for j in non_spurious:
            # Constrói silogismo sintético para validação
            major = f"Universal: {j.proposicao}"
            minor = f"Singular: {j.proposicao}"
            conclusion = j.proposicao
            violations = self.validator.validate(major, minor, conclusion)
            penalty = len(violations) * 0.1
            base_score = kb_scores.get(j.proposicao[:30], j.prioridade)
            scored.append((j, max(0.0, base_score - penalty)))

        # 3. Aplica falseabilidade (Popper)
        with_falsifiability = self.popper.apply(scored)

        # 4. Remove falsificadas e reordena
        valid = [
            (j, score)
            for j, score, falsified in with_falsifiability
            if not falsified
        ]
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid
