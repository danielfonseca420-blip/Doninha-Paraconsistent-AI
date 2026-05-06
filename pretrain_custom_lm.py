from __future__ import annotations

"""
Pré-treinamento de um pequeno modelo de linguagem
=================================================

Fluxo:
  1. Usa o texto do README (artigo/resumo) como corpus inicial.
  2. Treina um tokenizador SentencePiece (BPE) se ainda não existir.
  3. Constrói um Dataset de LM (inputs + labels deslocados).
  4. Treina `EpistemicLanguageModel` com cross-entropy e AdamW.
  5. Salva pesos do modelo e reutiliza o tokenizador treinado.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from custom_tokenizer import SPConfig, train_sentencepiece, CustomSPTokenizer
from custom_lm_model import LMConfig, EpistemicLanguageModel, save_lm, generate_text
from corpus_utils import load_main_corpus


@dataclass
class TrainLMConfig:
    sp_config: SPConfig = field(default_factory=SPConfig)
    max_seq_len: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    save_dir: str = "checkpoints_lm"


class LMDataset(Dataset):
    """
    Dataset de linguagem causal: divide o fluxo de tokens em blocos
    de tamanho fixo e usa input_ids e labels deslocados em 1.
    """

    def __init__(self, token_ids: List[int], block_size: int) -> None:
        self.block_size = block_size
        # Trunca para múltiplo de block_size
        n = (len(token_ids) // block_size) * block_size
        self.data = token_ids[:n]

    def __len__(self) -> int:
        return max(len(self.data) // self.block_size - 1, 0)

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : end + 1], dtype=torch.long)
        return x, y


def ensure_tokenizer(config: TrainLMConfig) -> CustomSPTokenizer:
    model_file = f"{config.sp_config.model_prefix}.model"
    if not os.path.exists(model_file):
        # Treina o SentencePiece a partir do corpus principal (README + artigo DOCX)
        texts = load_main_corpus()
        tmp_corpus = "sp_corpus_tmp.txt"
        with open(tmp_corpus, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\r\n", "\n") + "\n")
        train_sentencepiece([tmp_corpus], config.sp_config)
        os.remove(tmp_corpus)
    return CustomSPTokenizer(model_prefix=config.sp_config.model_prefix)


def build_token_stream(tokenizer: CustomSPTokenizer) -> Tuple[List[int], List[int]]:
    """
    Constrói streams de tokens para treino e validação a partir do corpus principal.
    Usa divisão simples train/val em nível de documento.
    """
    texts = load_main_corpus()
    if len(texts) == 1:
        train_texts = texts
        val_texts = texts
    else:
        split = max(1, int(0.8 * len(texts)))
        train_texts = texts[:split]
        val_texts = texts[split:]

    def encode_all(lst: List[str]) -> List[int]:
        ids: List[int] = []
        for t in lst:
            ids.extend(tokenizer.encode(t, add_bos=True, add_eos=True))
        return ids

    return encode_all(train_texts), encode_all(val_texts)


def evaluate_lm(
    model: EpistemicLanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
) -> float:
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += float(loss.item())
            steps += 1
    avg_loss = total_loss / max(steps, 1)
    return avg_loss


def train_lm(config: TrainLMConfig) -> EpistemicLanguageModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ensure_tokenizer(config)

    train_ids, val_ids = build_token_stream(tokenizer)

    train_dataset = LMDataset(train_ids, block_size=config.max_seq_len)
    val_dataset = LMDataset(val_ids, block_size=config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    lm_config = LMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=config.max_seq_len,
    )
    model = EpistemicLanguageModel(lm_config).to(device)

    # Suporte simples a múltiplas GPUs via DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        ):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # (batch, seq, vocab)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss / max(config.grad_accum_steps, 1)
            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                if config.grad_clip is not None and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.item())
            steps += 1

        scheduler.step()
        avg_train_loss = total_loss / max(steps, 1)

        # Validação
        val_loss = evaluate_lm(
            model.module if isinstance(model, torch.nn.DataParallel) else model,
            val_loader,
            device,
            loss_fn,
        )
        ppl = torch.exp(torch.tensor(val_loss)).item()

        print(
            f"Epoch {epoch+1} - train loss: {avg_train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | ppl: {ppl:.2f}"
        )

        # Pequena geração de teste
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        prompt = "A inteligência artificial"
        sample = generate_text(base_model, tokenizer, prompt, max_new_tokens=40)
        print(f"Exemplo de geração: {sample}\n")

        # Checkpoint por época
        ckpt_path = os.path.join(config.save_dir, f"epistemic_lm_epoch{epoch+1}.pt")
        save_lm(base_model, ckpt_path)

    # retorna o último modelo (sem DataParallel)
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def main() -> None:
    config = TrainLMConfig()
    model = train_lm(config)
    save_path = "epistemic_lm.pt"
    save_lm(model, save_path)
    print(f"Modelo de linguagem salvo em '{save_path}'")


if __name__ == "__main__":
    main()

