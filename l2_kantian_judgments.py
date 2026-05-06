"""
CAMADA L2 — Tábua de Juízos Kantianos
======================================
Antes de qualquer cálculo estatístico o prompt é destrinchado nas
doze categorias da Tábua dos Juízos (Kritik der reinen Vernunft, §9).

Dimensões:
  Quantidade  → Universal | Particular | Singular
  Qualidade   → Afirmativo | Negativo | Infinito
  Relação     → Categórico | Hipotético | Disjuntivo
  Modalidade  → Problemático | Assertórico | Apodítico

Cada hipótese gerada recebe um peso de prioridade; o Juízo Singular
Afirmativo Assertórico tem prioridade máxima (é a resposta-alvo).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
from l1_concept_table import ConceptNode, ConceptTable
import re


# ─────────────────────────────────────────────────────────────────────────────
# Estruturas de dados
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KantianJudgment:
    """Uma proposição refinada segundo a tábua dos juízos."""
    quantidade:  str   # Universal | Particular | Singular
    qualidade:   str   # Afirmativo | Negativo | Infinito
    relacao:     str   # Categórico | Hipotético | Disjuntivo
    modalidade:  str   # Problemático | Assertórico | Apodítico
    proposicao:  str   # texto da hipótese
    prioridade:  float = 0.0  # 0.0 → 1.0  (1.0 = resposta-alvo)

    def __str__(self) -> str:
        return (
            f"[{self.quantidade}/{self.qualidade}/"
            f"{self.relacao}/{self.modalidade}] "
            f"(pri={self.prioridade:.2f}) {self.proposicao}"
        )


@dataclass
class SyntaxProfile:
    """
    Perfil sintático mínimo extraído do enunciado segundo a gramática
    (aproximação heurística baseada em listas inspiradas em grammar.txt).
    """
    quantifier_subject: Optional[str] = None   # "all", "some", "this", etc.
    quantifier_predicate: Optional[str] = None
    has_negation: bool = False
    has_infinite_like: bool = False           # construções do tipo "not-X"
    is_conditional: bool = False              # presença de "if", "then"
    is_disjunctive: bool = False              # presença de "or"
    modality_markers: Tuple[str, ...] = ()    # "can", "must", "might", etc.


# ─────────────────────────────────────────────────────────────────────────────
# Regras de prioridade entre modalidades (herança da "parte fraca")
# ─────────────────────────────────────────────────────────────────────────────
MODALIDADE_PESO = {
    "Apodítico":    1.0,
    "Assertórico":  0.7,
    "Problemático": 0.4,
}
QUANTIDADE_PESO = {
    "Singular":   1.0,
    "Particular": 0.6,
    "Universal":  0.3,
}
QUALIDADE_PESO = {
    "Afirmativo": 1.0,
    "Infinito":   0.6,
    "Negativo":   0.4,
}
RELACAO_PESO = {
    "Categórico":  1.0,
    "Hipotético":  0.7,
    "Disjuntivo":  0.5,
}


def _priority(j: KantianJudgment) -> float:
    return (
        MODALIDADE_PESO[j.modalidade]
        * QUANTIDADE_PESO[j.quantidade]
        * QUALIDADE_PESO[j.qualidade]
        * RELACAO_PESO[j.relacao]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Motor de geração de juízos
# ─────────────────────────────────────────────────────────────────────────────

class KantianJudgmentEngine:
    """
    Recebe um prompt e a lista de ConceptNodes extraídos por L1 e devolve
    as 12 hipóteses estruturadas segundo a tábua kantiana.
    """

    def __init__(self, concept_table: ConceptTable) -> None:
        self.ct = concept_table

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def refine(self, prompt: str, concepts: List[ConceptNode]) -> List[KantianJudgment]:
        """
        Gera as hipóteses kantianas para o prompt e as ordena por
        prioridade descendente.
        """
        subject, predicates = self._parse_prompt(prompt, concepts)
        syntax = self._analyze_syntax(prompt)
        judgments: List[KantianJudgment] = []

        for pred in predicates:
            antonym = self._antonym_of(pred, concepts)
            hypernym = self._hypernym_of(pred, concepts)

            # ── Juízo principal guiado pela gramática ───────────────────
            qt = self._infer_quantity(syntax)
            ql = self._infer_quality(syntax)
            rel = self._infer_relation(syntax)
            mod = self._infer_modality(syntax)

            base_prop = f"{subject} é {pred}"
            if syntax.has_negation and antonym:
                base_prop = f"{subject} não é {antonym}"

            judgments.append(self._make(qt, ql, rel, mod, base_prop))

            # ── Variações canônicas (mantidas, mas ancoradas em L1) ─────
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categórico", "Apodítico",
                f"Todo(a) {subject} com propriedade extrema é {pred}",
            ))
            judgments.append(self._make(
                "Particular", "Afirmativo", "Hipotético", "Problemático",
                f"Algum(a) {subject} pode ser {pred}",
            ))
            judgments.append(self._make(
                "Singular", "Afirmativo", "Categórico", "Assertórico",
                f"Este(a) {subject} específico é {pred}",
            ))

            judgments.append(self._make(
                "Singular", "Negativo", "Categórico", "Assertórico",
                f"Este(a) {subject} não é {antonym}" if antonym else
                f"Este(a) {subject} não possui a propriedade oposta a {pred}",
            ))
            judgments.append(self._make(
                "Singular", "Infinito", "Categórico", "Assertórico",
                f"Este(a) {subject} é não-{antonym}" if antonym else
                f"Este(a) {subject} é indeterminado em relação a {pred}",
            ))

            judgments.append(self._make(
                "Universal", "Afirmativo", "Hipotético", "Apodítico",
                f"Se {subject} possui condição X, então é {pred}",
            ))
            judgments.append(self._make(
                "Universal", "Afirmativo", "Disjuntivo", "Assertórico",
                f"{subject} é {pred} OU {antonym} OU intermediário"
                if antonym else f"{subject} é {pred} ou outra propriedade",
            ))

            judgments.append(self._make(
                "Singular", "Afirmativo", "Categórico", "Problemático",
                f"Este(a) {subject} pode ser {pred}?",
            ))
            judgments.append(self._make(
                "Singular", "Afirmativo", "Hipotético", "Assertórico",
                f"Este(a) {subject} é {pred} em razão das condições observadas",
            ))
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categórico", "Apodítico",
                f"{subject} deve ser {pred} quando condições necessárias presentes",
            ))

            # ── HIPÓTESES COM INTERMEDIÁRIOS (hiperonímia) ───────────────
            if hypernym:
                judgments.append(self._make(
                    "Singular", "Afirmativo", "Categórico", "Assertórico",
                    f"Este(a) {subject} pertence à categoria {hypernym}",
                ))
            if antonym:
                judgments.append(self._make(
                    "Singular", "Negativo", "Disjuntivo", "Assertórico",
                    f"Este(a) {subject} não é {pred} nem {antonym}: "
                    f"admite valor intermediário",
                ))

        # Calcula prioridades e ordena
        for j in judgments:
            j.prioridade = _priority(j)
        judgments.sort(key=lambda j: j.prioridade, reverse=True)
        return judgments

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make(qt, ql, rel, mod, prop) -> KantianJudgment:
        j = KantianJudgment(
            quantidade=qt, qualidade=ql, relacao=rel,
            modalidade=mod, proposicao=prop,
        )
        j.prioridade = _priority(j)
        return j

    @staticmethod
    def _parse_prompt(prompt: str, concepts: List[ConceptNode]) -> Tuple[str, List[str]]:
        """
        Extrai sujeito e predicados candidatos do prompt de forma simples.
        Em produção seria substituído por um parser sintático.
        """
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", prompt.lower())
        known = {c.term.lower() for c in concepts}
        subject = tokens[0] if tokens else "entidade"
        predicates = [t for t in tokens[1:] if t in known] or ["indeterminado"]
        return subject, predicates

    # ------------------------------------------------------------------ #
    # Análise sintática inspirada em grammar.txt                         #
    # ------------------------------------------------------------------ #

    def _analyze_syntax(self, prompt: str) -> SyntaxProfile:
        """
        Extrai um perfil sintático mínimo usando listas de palavras
        alinhadas aos capítulos de determiners, modals, negatives e
        conjunctions da grammar COBUILD.
        """
        text = prompt.lower()
        tokens = re.findall(r"[a-záàãâéêíóôõúüç]+", text)

        quant_all = {"all", "every", "each"}
        quant_some = {"some", "many", "several", "few", "a few"}
        quant_singular = {"this", "that", "these", "those", "a", "an", "one"}

        neg_markers = {"not", "no", "never", "none", "nothing", "nowhere"}
        infinite_patterns = {"not-", "non-"}

        cond_markers = {"if", "provided", "unless", "whenever", "as long as"}
        disj_markers = {"or", "either"}

        modal_poss = {"can", "could", "may", "might"}
        modal_necess = {"must", "have to", "need to", "should", "ought"}

        has_neg = any(tok in neg_markers for tok in tokens)
        has_inf = any(pat in text for pat in infinite_patterns)
        is_cond = any(tok in cond_markers for tok in tokens)
        is_disj = any(tok in disj_markers for tok in tokens)

        mods: list[str] = []
        for tok in tokens:
            if tok in modal_poss or tok in modal_necess:
                mods.append(tok)

        q_subj: Optional[str] = None
        q_pred: Optional[str] = None

        if tokens:
            first = tokens[0]
            if first in quant_all:
                q_subj = "all"
            elif first in quant_some:
                q_subj = "some"
            elif first in quant_singular:
                q_subj = "this"

        return SyntaxProfile(
            quantifier_subject=q_subj,
            quantifier_predicate=q_pred,
            has_negation=has_neg,
            has_infinite_like=has_inf,
            is_conditional=is_cond,
            is_disjunctive=is_disj,
            modality_markers=tuple(mods),
        )

    def _infer_quantity(self, syntax: SyntaxProfile) -> str:
        if syntax.quantifier_subject == "all":
            return "Universal"
        if syntax.quantifier_subject == "some":
            return "Particular"
        if syntax.quantifier_subject == "this":
            return "Singular"
        return "Singular"

    def _infer_quality(self, syntax: SyntaxProfile) -> str:
        if syntax.has_infinite_like:
            return "Infinito"
        if syntax.has_negation:
            return "Negativo"
        return "Afirmativo"

    def _infer_relation(self, syntax: SyntaxProfile) -> str:
        if syntax.is_conditional:
            return "Hipotético"
        if syntax.is_disjunctive:
            return "Disjuntivo"
        return "Categórico"

    def _infer_modality(self, syntax: SyntaxProfile) -> str:
        markers = {m for m in syntax.modality_markers}
        if any(m in {"must", "have", "need", "should", "ought"} for m in markers):
            return "Apodítico"
        if any(m in {"can", "could", "may", "might"} for m in markers):
            return "Problemático"
        return "Assertórico"

    def _antonym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.antonyms:
            return node.antonyms[0]
        return ""

    def _hypernym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.hypernyms:
            return node.hypernyms[0]
        return ""
