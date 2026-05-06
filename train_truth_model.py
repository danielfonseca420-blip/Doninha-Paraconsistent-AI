from __future__ import annotations

"""
SCRIPT DE TREINAMENTO — TruthScoringModel
=========================================

Treina o modelo neural de avaliação de verdade (TruthScoringModel) com:

  1) Conjunto de regras do sistema paraconsistente (data/Fuzzy.txt):
     Gc = μ−λ, Gct = μ+λ−1, 12 estados, limites Vscc/Vicc/Vscct/Vicct.
     Gera dados sintéticos (μ, λ) → estado + valor-verdade para treinar L3.

  2) Opcional: documento DOCX com proposições (pré-treinamento fraco).

  - estado lógico: Verdadeiro | Falso | Intermediário | Indeterminado
  - valor-verdade escalar em [0,1]
"""

from dataclasses import dataclass
from typing import List, Tuple
import os
import re

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from neural_truth_model import (
    PropositionExample,
    PropositionDataset,
    TruthScoringModel,
    load_tokenizer,
)
from paraconsistent_rules import (
    load_rules_from_fuzzy_file,
    get_rules_training_examples,
    state_12_to_simple,
)

try:
    from corpus_utils import read_docx_file
except Exception:
    read_docx_file = None


@dataclass
class TrainConfig:
    backbone_name: str = "bert-base-multilingual-cased"
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 64
    use_fuzzy_rules: bool = True   # Treinar com conjunto de regras do Fuzzy.txt
    fuzzy_grid_step: float = 0.1   # Passo da grade (μ, λ) para dados sintéticos
    fuzzy_data_path: str | None = None  # Caminho para data/Fuzzy.txt (None = auto)


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def load_training_data_from_fuzzy_rules(
    config: TrainConfig,
) -> Tuple[List[PropositionExample], List[PropositionExample]]:
    """
    Gera dados de treino a partir do conjunto de regras do sistema paraconsistente
    estabelecido em data/Fuzzy.txt (LPA: Gc, Gct, 12 estados, para-analisador).
    Cada ponto (μ, λ) da grade é rotulado com estado e valor-verdade conforme as regras.
    """
    rules = load_rules_from_fuzzy_file(config.fuzzy_data_path)
    pairs = get_rules_training_examples(rules=rules, grid_step=config.fuzzy_grid_step)

    examples: List[PropositionExample] = []
    for mu, lam, state_12, truth in pairs:
        label_4 = state_12_to_simple(state_12)
        text = (
            f"Proposição com grau de crença {mu:.2f} e grau de descrença {lam:.2f}. "
            f"Certeza e contradição segundo análise paraconsistente."
        )
        examples.append(
            PropositionExample(
                text=text,
                label_state=label_4,
                truth_value=truth,
            )
        )

    if not examples:
        raise RuntimeError("Nenhum exemplo gerado a partir das regras do Fuzzy.txt.")
    split = max(1, int(0.8 * len(examples)))
    train = examples[:split]
    val = examples[split:]
    return train, val


def load_training_data_docx() -> Tuple[List[PropositionExample], List[PropositionExample]]:
    """
    Usa o artigo DOCX como fonte de proposições (pré-treinamento fraco).
    Todas marcadas como Indeterminado com truth_value 0.5.
    """
    if read_docx_file is None:
        raise RuntimeError("corpus_utils.read_docx_file não disponível.")
    base_dir = os.path.dirname(__file__) or "."
    article_path = os.path.join(
        base_dir,
        "Uma verdadeira Epistemologia para a Inteligência Artificial.docx",
    )
    text = read_docx_file(article_path)
    sentences = _split_sentences(text)

    examples: List[PropositionExample] = []
    for s in sentences:
        examples.append(
            PropositionExample(
                text=s,
                label_state="Indeterminado",
                truth_value=0.5,
            )
        )

    if not examples:
        raise RuntimeError("Nenhuma sentença extraída do artigo DOCX.")
    split = max(1, int(0.8 * len(examples)))
    train = examples[:split]
    val = examples[split:]
    return train, val


def load_training_data(config: TrainConfig) -> Tuple[List[PropositionExample], List[PropositionExample]]:
    """
    Carrega dados de treino: por padrão usa o conjunto de regras do Fuzzy.txt (L3 paraconsistente).
    Se use_fuzzy_rules=False, tenta carregar do DOCX.
    """
    if config.use_fuzzy_rules:
        return load_training_data_from_fuzzy_rules(config)
    return load_training_data_docx()


def train(config: TrainConfig) -> TruthScoringModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(config.backbone_name)
    train_examples, val_examples = load_training_data(config)

    train_dataset = PropositionDataset(train_examples, tokenizer, max_length=config.max_length)
    val_dataset = PropositionDataset(val_examples, tokenizer, max_length=config.max_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    model = TruthScoringModel(backbone_name=config.backbone_name).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels_state=batch["labels_state"],
                labels_truth=batch["labels_truth"],
            )
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} - train loss: {avg_loss:.4f}")

        # Validação simples (acurácia de estado)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = out["logits_state"].argmax(dim=-1)
                correct += int((preds == batch["labels_state"]).sum().item())
                total += int(batch["labels_state"].size(0))
        acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} - val acc (state): {acc:.4f}")

    return model


def main() -> None:
    config = TrainConfig(use_fuzzy_rules=True)
    print("Treinando L3 com conjunto de regras do sistema paraconsistente (data/Fuzzy.txt).")
    model = train(config)
    torch.save(model.state_dict(), "truth_scoring_model.pt")
    print("Modelo treinado salvo em 'truth_scoring_model.pt'")


if __name__ == "__main__":
    main()

