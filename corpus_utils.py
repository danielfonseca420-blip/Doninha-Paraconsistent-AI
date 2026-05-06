from __future__ import annotations

"""
Utilitários de corpus
=====================

Funções para carregar texto de:
  - arquivos Markdown/TXT (ex.: README)
  - artigo completo em DOCX ("Uma verdadeira Epistemologia para a Inteligência Artificial")
"""

from typing import List
import os

from docx import Document


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de texto não encontrado: {path}")
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def read_docx_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo DOCX não encontrado: {path}")
    doc = Document(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def load_main_corpus() -> List[str]:
    """
    Carrega o corpus principal deste projeto, incluindo materiais do projeto e bases de dados suplementares.

    Retorna uma lista de textos (documentos).
    """
    base_dir = os.path.dirname(__file__) or "."
    readme_path = os.path.join(base_dir, "README.md")
    article_path = os.path.join(
        base_dir,
        "Uma verdadeira Epistemologia para a Inteligência Artificial.docx",
    )

    extra_paths = [
        os.path.join(base_dir, "data", "stanford_encyclopedia", "sep_texts_only.txt"),
        os.path.join(base_dir, "philosophy-corpus", "train_philosophy.txt"),
        os.path.join(base_dir, "philosophy-corpus", "train.txt"),
    ]

    texts: List[str] = []
    if os.path.exists(readme_path):
        texts.append(read_text_file(readme_path))
    if os.path.exists(article_path):
        texts.append(read_docx_file(article_path))

    for path in extra_paths:
        if os.path.exists(path):
            texts.append(read_text_file(path))

    if not texts:
        raise FileNotFoundError(
            "Nenhum corpus encontrado. Certifique-se de que README.md, o artigo DOCX ou os arquivos da base de dados estão disponíveis."
        )
    return texts

