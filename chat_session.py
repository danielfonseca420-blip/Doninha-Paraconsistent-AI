"""
Sessão de chat com histórico.
=============================
Mantém as últimas N trocas (usuário/assistente) e monta o contexto
para o pipeline ou para o gerador.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


class ChatSession:
    """
    Histórico de mensagens para diálogo multi-turno.
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max(1, max_turns)
        self.turns: List[Turn] = []

    def add_user(self, content: str) -> None:
        self.turns.append(Turn(role="user", content=content.strip()))

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn(role="assistant", content=content.strip()))

    def get_context_for_prompt(self, current_prompt: str, max_turns_in_context: Optional[int] = None) -> str:
        """
        Retorna um único texto com as últimas N trocas + pergunta atual,
        para ser usado como contexto (ex.: prefixo da pergunta ou resumo).
        """
        n = max_turns_in_context if max_turns_in_context is not None else self.max_turns
        n = max(0, n)
        recent = self.turns[-n * 2 :] if n else []  # pares user/assistant
        parts = []
        for t in recent:
            prefix = "Usuário" if t.role == "user" else "Assistente"
            parts.append(f"{prefix}: {t.content}")
        if parts:
            parts.append(f"Usuário: {current_prompt}")
            return "\n".join(parts)
        return current_prompt

    def get_last_user_prompt(self) -> str:
        """Retorna a última mensagem do usuário (para pipeline que não usa contexto)."""
        for t in reversed(self.turns):
            if t.role == "user":
                return t.content
        return ""

    def clear(self) -> None:
        self.turns.clear()

    def turn_count(self) -> int:
        return len(self.turns)
