from __future__ import annotations

"""
Modelo de Linguagem Customizado (TransformerEncoder + BPE)
==========================================================

Pequeno modelo de linguagem causal baseado em TransformerEncoder, usando
tokens produzidos pelo `CustomSPTokenizer` (SentencePiece).

Funções principais:
  - `EpistemicLanguageModel`: arquitetura PyTorch
  - `generate_text`: função de geração autoregressiva
  - helpers para salvar/carregar pesos
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1


class EpistemicLanguageModel(nn.Module):
    """
    Modelo de linguagem simples (causal) com TransformerEncoder.
    """

    def __init__(self, config: LMConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        # Máscara causal: cada posição só vê tokens anteriores
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) com 1 para tokens válidos, 0 para pad
            # A API de TransformerEncoder usa src_key_padding_mask com True = pad
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        hidden = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        logits = self.lm_head(hidden)
        return logits


def generate_text(
    model: EpistemicLanguageModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> str:
    """
    Geração autoregressiva simples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.config.max_seq_len:
            break

        with torch.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab)
            next_token_logits = logits[0, -1, :] / max(temperature, 1e-4)

            if top_k > 0:
                values, indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)))
                probs = F.softmax(values, dim=-1)
                next_token = indices[torch.multinomial(probs, num_samples=1)]
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

    generated_ids: List[int] = input_ids[0].tolist()
    return tokenizer.decode(generated_ids)


def save_lm(model: EpistemicLanguageModel, path: str) -> None:
    torch.save({"config": model.config.__dict__, "state_dict": model.state_dict()}, path)


def load_lm(path: str, vocab_size: int) -> EpistemicLanguageModel:
    data = torch.load(path, map_location="cpu")
    cfg_dict = data.get("config", {})
    cfg_dict["vocab_size"] = vocab_size  # garante compatibilidade
    config = LMConfig(**cfg_dict)
    model = EpistemicLanguageModel(config)
    model.load_state_dict(data["state_dict"])
    return model

