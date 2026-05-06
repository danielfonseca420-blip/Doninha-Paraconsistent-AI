from __future__ import annotations

"""
Tokenizador SentencePiece (BPE/Unigram) customizado
===================================================

Treina um modelo SentencePiece a partir de um corpus de texto (por exemplo,
o texto do artigo/README) e expõe uma interface simples de encode/decode
para ser usada pelo modelo de linguagem customizado.
"""

from dataclasses import dataclass
from typing import List
import os

import sentencepiece as spm


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


@dataclass
class SPConfig:
    model_prefix: str = "sp_epistemologia"
    vocab_size: int = 2000
    model_type: str = "bpe"  # ou "unigram"


def train_sentencepiece(
    input_files: List[str],
    config: SPConfig = SPConfig(),
) -> None:
    """
    Treina um modelo SentencePiece a partir de uma lista de arquivos de texto.
    Gera `config.model_prefix.model` e `.vocab` na pasta atual.
    """
    input_str = ",".join(input_files)
    user_defined_symbols = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=config.model_prefix,
        vocab_size=config.vocab_size,
        model_type=config.model_type,
        character_coverage=0.9995,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        user_defined_symbols=user_defined_symbols,
    )


class CustomSPTokenizer:
    """
    Wrapper simples em torno de SentencePieceProcessor.

    Convenções:
      - `<bos>` é adicionado no início da sequência.
      - `<eos>` é adicionado no final (opcional).
    """

    def __init__(self, model_prefix: str = "sp_epistemologia") -> None:
        model_file = f"{model_prefix}.model"
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Modelo SentencePiece '{model_file}' não encontrado. "
                f"Treine primeiro com train_sentencepiece()."
            )
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        # Mapeia ids das user_defined_symbols
        self.pad_id = self.sp.piece_to_id("<pad>")
        self.bos_id = self.sp.piece_to_id("<bos>")
        self.eos_id = self.sp.piece_to_id("<eos>")

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        pieces = self.sp.encode(text, out_type=int)
        ids: List[int] = []
        if add_bos and self.bos_id >= 0:
            ids.append(self.bos_id)
        ids.extend(pieces)
        if add_eos and self.eos_id >= 0:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: List[int]) -> str:
        # remove tokens especiais se presentes
        filtered = [
            i
            for i in ids
            if i not in {self.bos_id, self.eos_id, self.pad_id} and i >= 0
        ]
        return self.sp.decode(filtered)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

