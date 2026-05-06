"""
Camada L5 — Geração de resposta em texto livre.
================================================
A partir da síntese L4 (e contexto L1–L3), gera resposta natural via LLM externo
(Groq) ou fallback para o template da L4. Opcional: LM customizado (EpistemicLanguageModel).
"""

from __future__ import annotations
import os
from typing import Optional

# Resultado da L4
try:
    from l4_synthesis import SynthesisResult
except Exception:
    SynthesisResult = None  # type: ignore


def build_context_for_generation(
    prompt: str,
    synthesis_result: "SynthesisResult",
    concepts_summary: str = "",
    top_judgments: str = "",
) -> str:
    """Monta o contexto (texto) a ser enviado ao LLM para gerar a resposta final."""
    lines = [
        "## Contexto epistemológico (L1–L4)",
        f"Pergunta do usuário: {prompt}",
        "",
        f"Resposta sintetizada (L4): {synthesis_result.response}",
        f"Valor de verdade: {synthesis_result.truth_value:.2f} | Estado: {synthesis_result.state} | Certeza: {synthesis_result.certainty:+.2f}",
        "",
    ]
    if synthesis_result.supporting_evidence:
        lines.append("Evidências de suporte:")
        for ev in synthesis_result.supporting_evidence[:5]:
            lines.append(f"  - {ev}")
        lines.append("")
    if concepts_summary:
        lines.append("Conceitos extraídos (L1):")
        lines.append(concepts_summary)
        lines.append("")
    if top_judgments:
        lines.append("Juízos relevantes (L2):")
        lines.append(top_judgments)
        lines.append("")
    lines.append("## Instrução")
    lines.append("Com base no contexto acima, elabore uma resposta final clara e precisa em português, sem repetir literalmente o texto da síntese. Seja conciso e cite a confiança quando relevante.")
    return "\n".join(lines)


def generate_with_groq(
    context: str,
    api_key: Optional[str] = None,
    model: str = "mixtral-8x7b-32768",
) -> str:
    """Gera resposta usando ChatGroq."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage
        llm = ChatGroq(model=model, api_key=api_key, temperature=0.3)
        msg = llm.invoke([HumanMessage(content=context)])
        return msg.content if hasattr(msg, "content") else str(msg)
    except Exception:
        return ""


def generate_with_custom_lm(
    context: str,
    model_path: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
) -> str:
    """Gera resposta usando EpistemicLanguageModel (custom_lm_model)."""
    try:
        from custom_lm_model import EpistemicLanguageModel, LMConfig, generate_text, load_lm
        from custom_tokenizer import CustomSPTokenizer, SPConfig
        import torch
        tokenizer = CustomSPTokenizer(SPConfig())
        tokenizer.load()
        vocab_size = tokenizer.vocab_size()
        model = load_lm(model_path, vocab_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        out = generate_text(model, tokenizer, context, max_new_tokens=max_new_tokens, temperature=temperature, device=device)
        return out or ""
    except Exception:
        return ""


def generate_response(
    prompt: str,
    synthesis_result: "SynthesisResult",
    provider: str = "template",
    concepts_summary: str = "",
    top_judgments: str = "",
    groq_model: str = "mixtral-8x7b-32768",
    custom_lm_path: str = "",
) -> str:
    """
    Gera a resposta final em texto livre (ou template).
    provider: "groq" | "template" | "custom_lm"
    """
    context = build_context_for_generation(prompt, synthesis_result, concepts_summary, top_judgments)

    if provider == "groq":
        text = generate_with_groq(context, model=groq_model)
        if text:
            return text.strip()

    if provider == "custom_lm" and custom_lm_path:
        text = generate_with_custom_lm(context, custom_lm_path)
        if text:
            return text.strip()

    # Fallback: resposta da L4 (template)
    return synthesis_result.response
