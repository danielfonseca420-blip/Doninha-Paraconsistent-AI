"""
Script de avaliação do pipeline.
================================
Executa o pipeline em um dataset de (pergunta, resposta_referência) e
calcula métricas (coerência L3, BLEU, ROUGE-L, opcionalmente similaridade).
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# Garante que o projeto está no path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_eval_dataset(path: str | Path) -> list:
    """Carrega dataset de eval: lista de dicts com prompt, reference_answer, etc."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def run_eval(
    dataset_path: str | Path | None = None,
    config_path: str | Path | None = None,
    verbose: bool = False,
) -> dict:
    """
    Roda o pipeline em cada item do dataset e agrega métricas.
    Retorna dict com scores médios e por exemplo.
    """
    from pipeline import HybridLLMPipeline
    from knowledge_base import get_knowledge_base
    from config_loader import load_config, PROJECT_ROOT
    from metrics import evaluate_response

    config = load_config(config_path)
    dataset_path = dataset_path or PROJECT_ROOT / "data" / "eval" / "sample.json"
    dataset = load_eval_dataset(dataset_path)
    if not dataset:
        return {"error": "Dataset vazio ou não encontrado", "path": str(dataset_path)}

    # Pipeline com KB carregado por config (sem RAG na eval para reprodutibilidade)
    kb = get_knowledge_base(config=config, config_path=config_path, query_for_rag=None)
    pipeline = HybridLLMPipeline(knowledge_base=kb, config=config, verbose=verbose)

    results = []
    all_bleu = []
    all_rouge = []
    all_coherence = []

    for item in dataset:
        prompt = item.get("prompt", "")
        reference = item.get("reference_answer", "")
        if not prompt:
            continue
        try:
            result = pipeline.process(prompt)
        except Exception as e:
            results.append({"id": item.get("id"), "error": str(e)})
            continue

        metrics = evaluate_response(result, reference_answer=reference if reference else None)
        results.append({
            "id": item.get("id"),
            "prompt": prompt[:80],
            "truth_value": result.truth_value,
            "coherence": metrics.get("coherence", {}),
            "bleu": metrics.get("bleu"),
            "rouge_l": metrics.get("rouge_l"),
            "semantic_similarity": metrics.get("semantic_similarity"),
        })
        if "coherence" in metrics and "coherence_score" in metrics["coherence"]:
            all_coherence.append(metrics["coherence"]["coherence_score"])
        if metrics.get("bleu") is not None:
            all_bleu.append(metrics["bleu"])
        if metrics.get("rouge_l") is not None:
            all_rouge.append(metrics["rouge_l"])

    out = {
        "n_examples": len(dataset),
        "n_processed": len(results),
        "results": results,
        "averages": {
            "coherence": sum(all_coherence) / len(all_coherence) if all_coherence else 0,
            "bleu": sum(all_bleu) / len(all_bleu) if all_bleu else 0,
            "rouge_l": sum(all_rouge) / len(all_rouge) if all_rouge else 0,
        },
    }
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Avaliação do pipeline L1–L4")
    parser.add_argument("--dataset", default=None, help="Caminho para JSON de eval")
    parser.add_argument("--config", default=None, help="Caminho para config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Log do pipeline")
    parser.add_argument("--output", default=None, help="Salvar resultado em JSON")
    args = parser.parse_args()

    out = run_eval(
        dataset_path=args.dataset,
        config_path=args.config,
        verbose=args.verbose,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Resultado salvo em {args.output}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
