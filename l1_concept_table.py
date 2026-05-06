"""
CAMADA L1 — Tábua de Conceitos (Aristóteles: Categorias)
=========================================================
Mapeia cada termo do prompt a relações semânticas fixas:
  - Sinonímia   : mesma denotação
  - Antonímia   : oposição semântica direta
  - Hiponímia   : relação específico → geral
  - Homonímia   : mesma forma, sentidos distintos
  - Paronímia   : semelhança formal, sentidos distintos

As relações são BINÁRIAS nesta camada — elimina a necessidade de
defuzzificação posterior na camada L3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re
import json
import os


@dataclass
class ConceptNode:
    """Um conceito na tábua, com todas as suas relações."""
    term: str
    definition: str = ""
    synonyms:   List[str] = field(default_factory=list)
    antonyms:   List[str] = field(default_factory=list)
    hyponyms:   List[str] = field(default_factory=list)   # mais específicos
    hypernyms:  List[str] = field(default_factory=list)   # mais gerais
    homonyms:   Dict[str, str] = field(default_factory=dict)  # sentido → definição
    paronyms:   List[str] = field(default_factory=list)
    domain:     str = "geral"


class ConceptTable:
    """
    Tábua de conceitos fixos.  Em produção seria alimentada por um
    dicionário / ontologia formal (WordNet-PT, OpenWordNet-PT, etc.).
    Aqui usamos um conjunto seminal suficiente para demonstrar todas
    as camadas do modelo.
    """

    def __init__(self) -> None:
        self._table: Dict[str, ConceptNode] = {}
        # Tábua seminal em português
        self._build_seed_table()
        # Banco de conceitos em inglês aprendido de dicionário externo (se existir)
        self._load_external_concepts()

    # ------------------------------------------------------------------ #
    # API pública                                                          #
    # ------------------------------------------------------------------ #

    def get(self, term: str) -> Optional[ConceptNode]:
        return self._table.get(self._normalize(term))

    def extract_concepts(self, text: str) -> List[ConceptNode]:
        """Extrai e retorna os nós de todos os termos encontrados no texto."""
        tokens = re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", text)
        seen, result = set(), []
        for tok in tokens:
            key = self._normalize(tok)
            if key not in seen:
                node = self._table.get(key)
                if node:
                    seen.add(key)
                    result.append(node)
        return result

    def add(self, node: ConceptNode) -> None:
        self._table[self._normalize(node.term)] = node

    def relation_type(self, term_a: str, term_b: str) -> str:
        """Retorna o tipo de relação semântica entre dois termos."""
        a = self._normalize(term_a)
        b = self._normalize(term_b)
        node_a = self._table.get(a)
        if not node_a:
            return "desconhecida"
        if b in [self._normalize(s) for s in node_a.synonyms]:
            return "sinonímia"
        if b in [self._normalize(s) for s in node_a.antonyms]:
            return "antonímia"
        if b in [self._normalize(s) for s in node_a.hyponyms]:
            return "hiponímia"
        if b in [self._normalize(s) for s in node_a.hypernyms]:
            return "hiperonímia"
        if b in [self._normalize(s) for s in node_a.paronyms]:
            return "paronímia"
        if b in [self._normalize(k) for k in node_a.homonyms]:
            return "homonímia"
        return "sem_relação_direta"

    # ------------------------------------------------------------------ #
    # Construção da tábua seminal                                          #
    # ------------------------------------------------------------------ #

    def _build_seed_table(self) -> None:
        entries = [
            ConceptNode(
                term="quente",
                definition="Que possui temperatura alta.",
                synonyms=["aquecido", "cálido", "morno", "tépido"],
                antonyms=["frio", "gelado", "fresco"],
                hypernyms=["temperatura"],
                hyponyms=["escaldante", "ardente"],
                domain="físico",
            ),
            ConceptNode(
                term="frio",
                definition="Que possui temperatura baixa.",
                synonyms=["gelado", "fresco", "frígido"],
                antonyms=["quente", "aquecido", "cálido"],
                hypernyms=["temperatura"],
                hyponyms=["congelado", "glacial"],
                domain="físico",
            ),
            ConceptNode(
                term="morno",
                definition="Entre quente e frio; tépido.",
                synonyms=["tépido", "ameno"],
                antonyms=["escaldante", "glacial"],
                hypernyms=["temperatura", "quente", "frio"],
                hyponyms=[],
                domain="físico",
            ),
            ConceptNode(
                term="temperatura",
                definition="Grandeza física que mede o grau de calor de um corpo.",
                synonyms=["calor", "grau"],
                antonyms=[],
                hypernyms=["grandeza_física"],
                hyponyms=["quente", "frio", "morno"],
                domain="físico",
            ),
            ConceptNode(
                term="água",
                definition="Substância H2O, geralmente em estado líquido.",
                synonyms=["H2O", "líquido"],
                antonyms=[],
                hypernyms=["substância", "fluido"],
                hyponyms=["vapor", "gelo"],
                domain="físico",
            ),
            ConceptNode(
                term="verdadeiro",
                definition="Que está de acordo com os fatos ou a realidade.",
                synonyms=["correto", "real", "factual"],
                antonyms=["falso", "incorreto", "fictício"],
                hypernyms=["valor_lógico"],
                domain="lógica",
            ),
            ConceptNode(
                term="falso",
                definition="Que não corresponde aos fatos ou à realidade.",
                synonyms=["incorreto", "errado", "fictício"],
                antonyms=["verdadeiro", "correto", "real"],
                hypernyms=["valor_lógico"],
                domain="lógica",
            ),
            ConceptNode(
                term="banco",
                definition="Móvel para sentar; instituição financeira; repositório de dados.",
                synonyms=[],
                antonyms=[],
                hypernyms=[],
                homonyms={
                    "assento": "móvel para sentar",
                    "financeiro": "instituição financeira",
                    "dados": "repositório de dados",
                },
                domain="geral",
            ),
            ConceptNode(
                term="eminente",
                definition="Pessoa ilustre ou notável.",
                synonyms=["ilustre", "notável"],
                antonyms=[],
                paronyms=["iminente"],
                domain="geral",
            ),
            ConceptNode(
                term="iminente",
                definition="Que está prestes a acontecer.",
                synonyms=["próximo", "imediato"],
                antonyms=[],
                paronyms=["eminente"],
                domain="geral",
            ),
            ConceptNode(
                term="inteligência",
                definition="Capacidade de compreender, raciocinar e resolver problemas.",
                synonyms=["cognição", "raciocínio", "entendimento"],
                antonyms=["ignorância", "estupidez"],
                hypernyms=["capacidade_mental"],
                domain="cognitivo",
            ),
            ConceptNode(
                term="conhecimento",
                definition="Ato ou efeito de conhecer; saber, ciência, erudição.",
                synonyms=["saber", "ciência", "erudição"],
                antonyms=["ignorância", "desconhecimento"],
                hypernyms=["epistemologia"],
                domain="filosófico",
            ),
            ConceptNode(
                term="verdade",
                definition="Conformidade entre o que se diz e o que é.",
                synonyms=["veracidade", "factualidade", "realidade"],
                antonyms=["mentira", "falsidade", "ilusão"],
                hypernyms=["epistemologia"],
                domain="filosófico",
            ),
        ]
        for node in entries:
            self.add(node)

    @staticmethod
    def _normalize(term: str) -> str:
        return term.strip().lower()

    # ------------------------------------------------------------------ #
    # Carregamento de conceitos externos (ex.: dicionário em inglês)      #
    # ------------------------------------------------------------------ #

    def _load_external_concepts(self) -> None:
        """
        Carrega conceitos adicionais de um banco gerado a partir do
        dicionário em inglês (arquivo JSON se existir).

        Formato esperado (lista de objetos):
          {
            "term": "abacus",
            "definition": "Frame with beads for calculating...",
            "synonyms": [],
            "antonyms": [],
            "hyponyms": [],
            "hypernyms": [],
            "domain": "geral"
          }
        """
        base_dir = os.path.dirname(__file__) or "."
        json_path = os.path.join(base_dir, "data", "concepts_en.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            return

        for item in items:
            term = item.get("term")
            if not term:
                continue
            node = ConceptNode(
                term=term,
                definition=item.get("definition", ""),
                synonyms=item.get("synonyms", []),
                antonyms=item.get("antonyms", []),
                hyponyms=item.get("hyponyms", []),
                hypernyms=item.get("hypernyms", []),
                homonyms=item.get("homonyms", {}),
                paronyms=item.get("paronyms", []),
                domain=item.get("domain", "geral"),
            )
            # Não sobrescreve conceitos portugueses existentes
            key = self._normalize(term)
            if key not in self._table:
                self._table[key] = node

