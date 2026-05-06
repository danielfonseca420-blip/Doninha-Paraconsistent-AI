from __future__ import annotations

"""
MÓDULO NEURAL — TruthScoringModel
=================================

Modelo PyTorch baseado em Transformer (via `transformers`) que recebe
proposições textuais (saída de L2) e produz:

  - logits de classe para o estado paraconsistente:
        Verdadeiro | Falso | Intermediário | Indeterminado
  - um escalar v ∈ [0,1] representando o valor-verdade aproximado
    (compatível com `ParaconsistentValue.truth_value`).

Este módulo NÃO é acoplado diretamente ao pipeline; ele pode ser
instanciado e passado opcionalmente para a `ParaconsistentEngine`,
que o usará para calcular (μ, λ) neurais em vez das heurísticas puras.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Rótulos paraconsistentes
# ─────────────────────────────────────────────────────────────────────────────

LABEL2ID: Dict[str, int] = {
    "Verdadeiro": 0,
    "Falso": 1,
    "Intermediário": 2,
    "Indeterminado": 3,
}

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


@dataclass
class PropositionExample:
    """Exemplo supervisionado para treinamento do modelo neural."""

    text: str
    label_state: str          # uma das chaves de LABEL2ID
    truth_value: float        # valor-verdade escalar em [0,1]


class PropositionDataset(Dataset):
    """
    Dataset simples de proposições rotuladas com estado paraconsistente
    e valor-verdade escalar.
    """

    def __init__(self, examples: List[PropositionExample], tokenizer, max_length: int = 64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels_state"] = torch.tensor(LABEL2ID[ex.label_state], dtype=torch.long)
        item["labels_truth"] = torch.tensor(float(ex.truth_value), dtype=torch.float)
        return item


class TruthScoringModel(nn.Module):
    """
    Modelo híbrido:
      - backbone TransformerEncoder (BERT-like)
      - cabeça de classificação para o estado lógico
      - cabeça de regressão para valor-verdade escalar
    """

    def __init__(
        self,
        backbone_name: str = "bert-base-multilingual-cased",
        num_labels: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Linear(hidden_size, num_labels)
        self.truth_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),  # restringe para [0,1]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_state: Optional[torch.Tensor] = None,
        labels_truth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]

        logits_state = self.classifier(cls_emb)
        truth_score = self.truth_head(cls_emb).squeeze(-1)

        loss: Optional[torch.Tensor] = None
        if labels_state is not None and labels_truth is not None:
            ce_loss = nn.CrossEntropyLoss()(logits_state, labels_state)
            mse_loss = nn.MSELoss()(truth_score, labels_truth)
            loss = ce_loss + 0.5 * mse_loss

        return {
            "logits_state": logits_state,
            "truth_score": truth_score,
            "loss": loss,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de inferência
# ─────────────────────────────────────────────────────────────────────────────

def load_tokenizer(backbone_name: str = "bert-base-multilingual-cased"):
    """Cria um tokenizer compatível com o backbone."""
    return AutoTokenizer.from_pretrained(backbone_name)


def score_proposition(
    model: TruthScoringModel,
    tokenizer,
    text: str,
    device: Optional[torch.device] = None,
) -> Tuple[str, float]:
    """
    Executa inferência para uma única proposição textual.
    Retorna (label_state, truth_score).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    model = model.to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out["logits_state"]
    truth_score = float(out["truth_score"].cpu().item())
    pred_id = int(logits.argmax(dim=-1).cpu().item())
    pred_label = ID2LABEL[pred_id]
    return pred_label, truth_score


def neural_annotations(
    model: TruthScoringModel,
    tokenizer,
    text: str,
) -> Tuple[float, float, str, float]:
    """
    Mapeia a saída do modelo neural para (μ, λ) compatíveis com L3.
    Retorna (mu, lam, state_label, truth_score).
    """
    state, v = score_proposition(model, tokenizer, text)
    if state == "Verdadeiro":
        mu = v
        lam = 1.0 - v
    elif state == "Falso":
        mu = 1.0 - v
        lam = v
    elif state == "Intermediário":
        mu = 0.4 + 0.2 * v
        lam = 0.4 + 0.2 * (1.0 - v)
    else:  # Indeterminado
        mu = 0.3
        lam = 0.3
    return float(mu), float(lam), state, float(v)

