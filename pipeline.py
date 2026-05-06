"""
PIPELINE PRINCIPAL — Modelo Híbrido de LLM
===========================================
Orquestra as 8 etapas do fluxo completo:

  1. Recepção do prompt
  2. Extração de conceitos [L1]
  3. Refinamento por Juízos Kantianos [L2]
  4. Silogismo Científico + Hempel
  5. Falseabilidade de Popper
  6. Avaliação Paraconsistente [L3]
  7. Síntese por Equivalência [L4]
  8. Geração da Resposta [L5 — opcional]

Usa config_loader, knowledge_base (KB escalável + RAG opcional), l5_generation
e opcionalmente o agente de pesquisa para enriquecer contexto.
"""

from __future__ import annotations
import sys
import re
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from neural_truth_model import TruthScoringModel, load_tokenizer
from l1_concept_table import ConceptTable, ConceptNode
from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment
from syllogism_module import ScientificSyllogismPipeline
from l3_paraconsistent import ParaconsistentEngine, ParaconsistentValue
from l4_synthesis import RussellianSynthesisEngine, SynthesisResult

try:
    from l4_russell_equivalence import load_concept_base
except Exception:
    load_concept_base = None  # type: ignore

try:
    from config_loader import load_config, PROJECT_ROOT
except Exception:
    load_config = None  # type: ignore
    PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from knowledge_base import get_knowledge_base, SEED_KNOWLEDGE_BASE
except Exception:
    get_knowledge_base = None  # type: ignore
    SEED_KNOWLEDGE_BASE = {}

try:
    from l5_generation import generate_response as l5_generate
except Exception:
    l5_generate = None  # type: ignore

try:
    from agente_busca_web import run_search_for_context
except Exception:
    run_search_for_context = None  # type: ignore


def _get_kb(config: Optional[Dict[str, Any]], prompt: str, use_agent: bool) -> Dict[str, float]:
    if get_knowledge_base is None:
        return dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
    return get_knowledge_base(
        config=config,
        query_for_rag=prompt if use_agent else None,
    )


class HybridLLMPipeline:
    """
    Pipeline completo do Modelo Híbrido de LLM.
    Suporta config, KB escalável, L5 (geração), agente opcional e chat.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self._config = config or (load_config() if load_config else {})
        self.kb = knowledge_base or _get_kb(self._config, "", False)
        if not self.kb:
            self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        self.verbose = verbose

        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine(self.L1)
        self.SYL = ScientificSyllogismPipeline()

        # L3
        l3_cfg = self._config.get("l3", {})
        model_path = l3_cfg.get("model_path", "truth_scoring_model.pt")
        backbone_name = l3_cfg.get("backbone", "bert-base-multilingual-cased")
        if not Path(model_path).is_absolute():
            model_path = str(PROJECT_ROOT / model_path)
        neural_model = None
        neural_tokenizer = None
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                neural_tokenizer = load_tokenizer(backbone_name)
                neural_model = TruthScoringModel(backbone_name=backbone_name)
                state = torch.load(model_path, map_location=device)
                neural_model.load_state_dict(state)
                neural_model.to(device)
                if self.verbose:
                    print(f"[L3] Modelo neural carregado de '{model_path}'")
                self.L3 = ParaconsistentEngine(neural_model=neural_model, neural_tokenizer=neural_tokenizer, device=device)
            except Exception as exc:
                if self.verbose:
                    print(f"[L3] Falha ao carregar modelo neural: {exc}")
                self.L3 = ParaconsistentEngine()
        else:
            self.L3 = ParaconsistentEngine()

        # L4
        russell_base = None
        rpath = self._config.get("l4", {}).get("russell_concepts_path", "l4_russell_concepts.json")
        if not Path(rpath).is_absolute():
            rpath = str(PROJECT_ROOT / rpath)
        if load_concept_base and os.path.exists(rpath):
            try:
                russell_base = load_concept_base(rpath)
                if self.verbose:
                    print("[L4] Base russelliana carregada.")
            except Exception:
                pass
        if russell_base is None and load_concept_base:
            try:
                from l4_russell_equivalence import build_russell_concept_base
                russell_base = build_russell_concept_base()
            except Exception:
                pass
        self.L4 = RussellianSynthesisEngine(
            self.kb,
            russell_concept_base=russell_base,
            use_concept_based_weights=(russell_base is not None),
        )

    def process(
        self,
        prompt: str,
        chat_session: Optional[Any] = None,
        use_agent: Optional[bool] = None,
        skip_l5: bool = False,
    ) -> SynthesisResult:
        """Executa o pipeline e retorna SynthesisResult (com response já gerada por L5 se ativo)."""
        t0 = time.perf_counter()
        use_agent = use_agent if use_agent is not None else self._config.get("agent", {}).get("use_agent", False)
        if chat_session and hasattr(chat_session, "get_context_for_prompt"):
            prompt_for_kb = chat_session.get_context_for_prompt(prompt, self._config.get("chat", {}).get("max_turns_in_context", 10))
        else:
            prompt_for_kb = prompt

        # KB pode ser enriquecido por RAG (Chroma) quando use_agent
        if use_agent and get_knowledge_base:
            self.kb = _get_kb(self._config, prompt_for_kb, True)
            if not self.kb:
                self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}

        self._log("\n" + "═" * 60)
        self._log(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        self._log("═" * 60)

        limit = RussellianSynthesisEngine.check_fundamental_limits(prompt)
        if limit:
            self._log(f"\n{limit}")

        self._log("\n[ETAPA 2] L1 — Extração de Conceitos")
        concepts: List[ConceptNode] = self.L1.extract_concepts(prompt)
        concepts_summary = ""
        if self.verbose and concepts:
            for c in concepts:
                syns = ", ".join(c.synonyms[:2]) or "—"
                self._log(f"  • {c.term:15s} | sinônimos: {syns}")
            concepts_summary = "; ".join(f"{c.term}({', '.join(c.synonyms[:2])})" for c in concepts[:8])

        self._log("\n[ETAPA 3] L2 — Juízos Kantianos")
        judgments: List[KantianJudgment] = self.L2.refine(prompt, concepts)
        top_judgments = ""
        if judgments:
            top_judgments = "\n".join(j.proposicao for j, _ in list(zip(judgments, [None] * 6))[:6])

        self._log("\n[ETAPAS 4+5] Silogismo + Hempel + Popper")
        prompt_terms = set(re.findall(r"[a-záàãâéêíóôõúüçA-ZÁÀÃÂÉÊÍÓÔÕÚÜÇ]+", prompt.lower()))
        kb_scores = {j.proposicao[:30]: self.kb.get(j.proposicao.split()[0], 0.3) for j in judgments}
        filtered = self.SYL.run(judgments, prompt_terms, kb_scores)
        self._log(f"  {len(judgments)} hipóteses → {len(filtered)} após filtros")

        self._log("\n[ETAPA 6] L3 — Lógica Paraconsistente")
        props_with_priority = [(j.proposicao, score) for j, score in filtered]
        pv_list: List[ParaconsistentValue] = self.L3.evaluate(props_with_priority, self.kb)
        consistent = self.L3.check_global_consistency(pv_list)
        self._log(f"  Consistência global: {'✓' if consistent else '✗'}")

        self._log("\n[ETAPA 7] L4 — Síntese Russelliana")
        l2_priorities = {j.proposicao[:40]: j.prioridade for j, _ in filtered}
        result: SynthesisResult = self.L4.synthesize(pv_list, l2_priorities, prompt)

        # Contexto do agente (busca web/local) se ativo
        agent_context = ""
        if use_agent and run_search_for_context:
            try:
                agent_context = run_search_for_context(prompt)
                if agent_context and self.verbose:
                    self._log("\n[AGENTE] Contexto de busca obtido.")
            except Exception:
                pass

        # L5 — Geração de resposta em texto livre
        gen_cfg = self._config.get("generation", {})
        provider = gen_cfg.get("provider", "template")
        if not skip_l5 and l5_generate and provider != "template":
            final_response = l5_generate(
                prompt,
                result,
                provider=provider,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                groq_model=gen_cfg.get("groq_model", "mixtral-8x7b-32768"),
                custom_lm_path=gen_cfg.get("custom_lm_path", ""),
            )
            if agent_context and final_response:
                final_response = final_response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            elif agent_context:
                final_response = result.response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            else:
                final_response = final_response or result.response
            result = SynthesisResult(
                response=final_response,
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
        elif agent_context and result.response:
            result = SynthesisResult(
                response=result.response + "\n\n[Contexto da busca]\n" + agent_context[:800],
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        self._log(f"\n[ETAPA 8] Resposta Final  ({elapsed:.1f} ms)\n")
        self._log(str(result))
        return result

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def repl(self) -> None:
        print("\n" + "═" * 60)
        print("  MODELO HÍBRIDO DE LLM — Fonseca")
        print("  Digite 'sair' para encerrar")
        print("═" * 60)
        while True:
            try:
                prompt = input("\nPrompt › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit"}:
                break
            self.process(prompt)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Modelo Híbrido de LLM — Pipeline L1–L5")
    parser.add_argument("--prompt", "-p", type=str, help="Pergunta única (imprime só a resposta)")
    parser.add_argument("--repl", action="store_true", help="Modo interativo")
    parser.add_argument("--demo", action="store_true", help="Rodar demonstração com prompts fixos")
    parser.add_argument("--config", type=str, help="Caminho para config.yaml")
    args, _ = parser.parse_known_args()

    config = load_config(Path(args.config)) if load_config and args.config else (load_config() if load_config else {})
    pipeline = HybridLLMPipeline(config=config, verbose=not args.prompt)

    if args.prompt:
        r = pipeline.process(args.prompt)
        print(r.response)
        return
    if args.repl:
        pipeline.repl()
        return
    if args.demo:
        for p in ["A água a 35 graus está quente ou fria?", "O que é a verdade?"]:
            pipeline.process(p)
            print()
        return
    # Default: demo + repl se --repl no argv antigo
    if "--repl" in sys.argv:
        pipeline.repl()
        return
    for p in ["A água a 35 graus está quente ou fria?", "O que é a verdade?"]:
        pipeline.process(p)
        print()


if __name__ == "__main__":
    main()
