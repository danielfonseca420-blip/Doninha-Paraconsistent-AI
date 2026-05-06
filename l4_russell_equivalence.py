"""
Base teórica russelliana para a camada L4 — Equivalência e Correspondência
============================================================================
Utiliza o arquivo data/russell.txt (Bertrand Russell, The Problems of Philosophy)
para fundamentar a síntese L4 no conceito de EQUIVALÊNCIA como correspondência
entre crença/proposição e fato, e não apenas em agregação estatística.

Conceitos extraídos do Cap. XII (Truth and Falsehood):
  - Verdade = correspondência entre crença e fato.
  - Fato = unidade complexa formada pelos objetos da crença na mesma ordem.
  - Crença verdadeira quando existe fato correspondente; falsa quando não existe.
  - Propriedade extrínseca: a verdade depende da relação da crença com algo externo.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ─── Conceitos russellianos (extraídos do texto) ───────────────────────────
# Termos que indicam alta relevância para equivalência/correspondência
CORRESPONDENCE_TERMS = [
    "correspondence", "correspond", "corresponding", "corresponds",
    "equivalence", "equivalent", "match", "accord", "agree", "fact",
    "belief", "true", "truth", "false", "falsehood", "beliefs", "facts",
    "object-terms", "object-relation", "complex unity", "constituents",
    "judgement", "judging", "sense-data", "physical object",
]
# Normalizados para matching em português/inglês
EQUIVALENCE_CONCEPTS_PT = [
    "correspondência", "equivalência", "crença", "fato", "verdade", "falsidade",
    "juízo", "objeto", "termos", "relação", "unidade", "complexo",
    "dado sensível", "proposição", "conhecimento",
]


@dataclass
class RussellConceptBase:
    """
    Base de conceitos extraída de russell.txt para fundamentar a síntese L4.
    Permite ponderar proposições por alinhamento teórico (correspondência com fatos)
    e não apenas por estatística.
    """
    # Trechos do texto sobre verdade/correspondência (cap. XII e adjacentes)
    key_passages: List[str] = field(default_factory=list)
    # Termos do texto com peso conceitual (relevância para equivalência)
    term_weights: Dict[str, float] = field(default_factory=dict)
    # Princípio em forma de texto (para auditoria/interpretação)
    principle_summary: str = ""

    def concept_weight_for_terms(self, terms: List[str]) -> float:
        """
        Peso conceitual para um conjunto de termos: quanto mais os termos
        aparecem na base russelliana, mais a proposição é tratada como
        alinhada à teoria da equivalência (correspondência crença–fato).
        """
        if not self.term_weights:
            return 1.0
        total = 0.0
        count = 0
        for t in terms:
            t_lower = t.lower().strip()
            if t_lower in self.term_weights:
                total += self.term_weights[t_lower]
                count += 1
        if count == 0:
            return 1.0
        return 1.0 + (total / count) * 0.5  # modulação suave


def _normalize_word(w: str) -> str:
    return re.sub(r"[^a-záàãâéêíóôõúüç0-9]", "", w.lower())


def load_russell_text(path: Optional[str] = None) -> str:
    """Carrega o conteúdo de data/russell.txt."""
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "data", "russell.txt")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_chapter_xii(content: str) -> str:
    """Extrai o capítulo XII (Truth and Falsehood) e trechos adjacentes relevantes."""
    start = content.find("CHAPTER XII")
    if start == -1:
        start = content.find("TRUTH AND FALSEHOOD")
    if start == -1:
        return content[:15000]  # fallback: início do livro
    end = content.find("CHAPTER XIV", start)
    if end == -1:
        end = content.find("CHAPTER XV", start)
    if end == -1:
        end = len(content)
    return content[start:end]


def extract_equivalence_passages(content: str) -> List[str]:
    """Extrai trechos que definem equivalência/correspondência."""
    chapter_xii = extract_chapter_xii(content)
    # Frases que contêm os conceitos centrais
    sentences = re.split(r"[.!?]\s+", chapter_xii)
    key = []
    for s in sentences:
        s_lower = s.lower()
        if any(
            x in s_lower
            for x in (
                "correspondence",
                "correspond",
                "belief",
                "fact",
                "true",
                "false",
                "complex unity",
                "object-terms",
                "object-relation",
            )
        ):
            key.append(s.strip())
    return key[:50]  # limite razoável


def build_term_weights_from_russell(content: str) -> Dict[str, float]:
    """
    Constrói pesos por termo a partir do texto de Russell: termos que aparecem
    em contextos de verdade/correspondência recebem peso maior.
    """
    chapter = extract_chapter_xii(content)
    words = re.findall(r"[a-záàãâéêíóôõúüç]+", chapter.lower())
    # Frequência no capítulo de verdade
    freq: Dict[str, int] = {}
    for w in words:
        w = _normalize_word(w)
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    # Normalizar para [0.2, 1.0] por relevância conceitual
    concept_set = set(
        _normalize_word(t) for t in CORRESPONDENCE_TERMS + EQUIVALENCE_CONCEPTS_PT
    )
    max_f = max(freq.values()) if freq else 1
    term_weights: Dict[str, float] = {}
    for w, c in freq.items():
        if w in concept_set:
            term_weights[w] = 0.5 + 0.5 * (c / max_f)
        else:
            term_weights[w] = 0.2 + 0.3 * (c / max_f)
    return term_weights


def build_russell_concept_base(path: Optional[str] = None) -> RussellConceptBase:
    """
    Treina/constroi a base de conceitos russellianos a partir de russell.txt.
    Usado pela L4 para síntese fundamentada em equivalência (correspondência).
    """
    content = load_russell_text(path)
    passages = extract_equivalence_passages(content)
    term_weights = build_term_weights_from_russell(content)
    summary = (
        "Truth consists in correspondence between belief and fact. "
        "A belief is true when there is a corresponding fact (complex unity of the objects of the belief). "
        "Truth and falsehood are extrinsic properties: they depend on the relation of the belief to outside things."
    )
    return RussellConceptBase(
        key_passages=passages,
        term_weights=term_weights,
        principle_summary=summary,
    )


def score_proposition_by_concepts(
    proposition: str,
    knowledge_base: Dict[str, float],
    concept_base: RussellConceptBase,
) -> float:
    """
    Score conceitual da proposição: grau em que ela se alinha à teoria da
    equivalência (correspondência com fatos/BD), não apenas estatística.

    - Termos da proposição que estão no KB com alta evidência indicam
      melhor "correspondência" com o mundo (fatos).
    - Termos que aparecem na base russelliana aumentam o peso teórico.
    """
    words = re.findall(r"[a-záàãâéêíóôõúüç]+", proposition.lower())
    terms = [_normalize_word(w) for w in words if len(w) > 2]

    # 1) Alinhamento com fatos (KB): termos da proposição presentes no BD
    kb_match = 0.0
    n = 0
    for t in terms:
        for kb_term, ev in knowledge_base.items():
            if _normalize_word(kb_term) == t or t in _normalize_word(kb_term):
                kb_match += ev
                n += 1
                break
    fact_alignment = (kb_match / n) if n > 0 else 0.5  # neutro se nenhum termo no KB

    # 2) Peso conceitual russelliano (termos da teoria)
    concept_weight = concept_base.concept_weight_for_terms(terms)

    # Combinação: correspondência com fatos (BD) + alinhamento teórico
    return (0.7 * fact_alignment + 0.3 * concept_weight)


def save_concept_base(base: RussellConceptBase, path: str) -> None:
    """Salva a base de conceitos para uso posterior da L4."""
    import json
    data = {
        "principle_summary": base.principle_summary,
        "key_passages": base.key_passages[:20],
        "term_weights": base.term_weights,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_concept_base(path: str) -> RussellConceptBase:
    """Carrega base de conceitos previamente construída."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RussellConceptBase(
        principle_summary=data.get("principle_summary", ""),
        key_passages=data.get("key_passages", []),
        term_weights=data.get("term_weights", {}),
    )
