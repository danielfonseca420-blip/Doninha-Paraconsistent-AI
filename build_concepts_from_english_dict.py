from __future__ import annotations

"""
Geração de banco de conceitos (L1) a partir do dicionário em inglês
===================================================================

Lê o arquivo de texto `data/English dictonary.txt` e constrói um
`concepts_en.json` com entradas básicas:

  - term
  - definition (linha principal do verbete)
  - domain aproximado (quando houver marcadores como 'naut.', 'archit.', etc.)

Este JSON é então carregado automaticamente por `ConceptTable._load_external_concepts`.
"""

from dataclasses import dataclass, asdict
from typing import List
import json
import os
import re


@dataclass
class EnglishConceptEntry:
    term: str
    definition: str
    synonyms: List[str]
    antonyms: List[str]
    hyponyms: List[str]
    hypernyms: List[str]
    homonyms: dict
    paronyms: List[str]
    domain: str = "geral"


DOMAIN_MARKERS = {
    "naut.": "náutico",
    "biol.": "biologia",
    "astron.": "astronomia",
    "archit.": "arquitetura",
    "med.": "medicina",
}


def guess_domain(line: str) -> str:
    for marker, domain in DOMAIN_MARKERS.items():
        if marker in line:
            return domain
    return "geral"


def parse_dictionary(path: str) -> List[EnglishConceptEntry]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de dicionário não encontrado: {path}")

    entries: List[EnglishConceptEntry] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Ignora cabeçalhos isolados ('A', 'B', etc.)
            if len(line) == 1 and line.isalpha():
                continue

            # Padrão aproximado: "Headword  rest of line"
            m = re.match(r"^([A-Za-z][A-Za-z0-9 .'-]*?)\s{2,}(.*)$", line)
            if not m:
                continue
            term, rest = m.group(1).strip(), m.group(2).strip()
            if not term or not rest:
                continue

            definition = rest
            domain = guess_domain(rest)

            entry = EnglishConceptEntry(
                term=term,
                definition=definition,
                synonyms=[],
                antonyms=[],
                hyponyms=[],
                hypernyms=[],
                homonyms={},
                paronyms=[],
                domain=domain,
            )
            entries.append(entry)

    return entries


def main() -> None:
    base_dir = os.path.dirname(__file__) or "."
    dict_path = os.path.join(base_dir, "data", "English dictonary.txt")
    out_path = os.path.join(base_dir, "data", "concepts_en.json")

    concepts = parse_dictionary(dict_path)
    data = [asdict(c) for c in concepts]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Gerado banco de conceitos com {len(concepts)} entradas em '{out_path}'")


if __name__ == "__main__":
    main()

