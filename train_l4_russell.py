"""
Treino da camada L4 a partir de russell.txt — Base teórica de equivalência
============================================================================
Constrói a base de conceitos russellianos (RussellConceptBase) a partir do
arquivo data/russell.txt e salva para uso pela L4.

Conceito central (Russell, The Problems of Philosophy, Cap. XII):
  - Verdade = correspondência entre crença e fato.
  - Equivalência (para a IA) = grau de correspondência entre a proposição
    refinada (L1–L3) e os "fatos" representados no banco de conhecimento.
  - A síntese L4 passa a usar um cálculo fundamentado em conceitos
    (correspondência crença–fato) e não somente análise estatística.

Uso:
  python train_l4_russell.py
  python train_l4_russell.py --data data/russell.txt --out l4_russell_concepts.json
"""

from __future__ import annotations
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Treina a base de conceitos russellianos da L4 a partir de russell.txt"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Caminho para russell.txt (default: data/russell.txt)",
    )
    parser.add_argument(
        "--out",
        default="l4_russell_concepts.json",
        help="Arquivo de saída da base de conceitos (default: l4_russell_concepts.json)",
    )
    args = parser.parse_args()

    try:
        from l4_russell_equivalence import (
            build_russell_concept_base,
            save_concept_base,
            load_russell_text,
            extract_chapter_xii,
        )
    except ImportError as e:
        print("Erro ao importar l4_russell_equivalence:", e, file=sys.stderr)
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(base_dir, "data", "russell.txt")

    if not os.path.isfile(data_path):
        print(f"Arquivo não encontrado: {data_path}", file=sys.stderr)
        sys.exit(1)

    print("Carregando russell.txt para fundamentar L4 em equivalência (correspondência crença–fato).")
    content = load_russell_text(data_path)
    print(f"  Cap. XII (Truth and Falsehood): {len(extract_chapter_xii(content))} caracteres.")

    base = build_russell_concept_base(data_path)
    print(f"  Passagens extraídas: {len(base.key_passages)}")
    print(f"  Termos com peso conceitual: {len(base.term_weights)}")

    out_path = args.out if os.path.isabs(args.out) else os.path.join(base_dir, args.out)
    save_concept_base(base, out_path)
    print(f"Base de conceitos russelliana salva em: {out_path}")
    print("L4 pode carregar com: load_concept_base(...) e passar para RussellianSynthesisEngine.")


if __name__ == "__main__":
    main()
