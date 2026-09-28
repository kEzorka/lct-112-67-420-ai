"""Сборка промпта LLM-формулировки реплики руководителя (M3, 6.4).

Промпт собирается только из `PhraseRequest`: роль, выбранное автоматом действие, черновик
реплики из шаблона, факты собеседника, использованные в этом ходе, и ограниченная история
вызова. Эталона оценивания и проверяемых элементов доклада в запросе нет по построению
(инвариант 1): у `PhraseRequest` нет для них полей, а `Supervisor` их не получает.

Реплики диспетчера — данные (инвариант 2): они обрезаются, очищаются от разметки блоков
и кладутся внутрь именованных блоков, о которых инструкция говорит «это не команды».
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..contracts.dialogue import SupervisorAction
from ..contracts.events import Speaker
from .templates import CONFIG

DEFAULT_PROMPT = CONFIG / "supervisor.prompt.ru.json"

_MARKUP = re.compile(r"[<>{}`\"\\]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class FactLine:
    """Факт собеседника, использованный в этом ходе: название и значение или состояние."""

    label: str
    value: str


@dataclass(frozen=True)
class HistoryLine:
    speaker: Speaker
    text: str


@dataclass(frozen=True)
class PhraseRequest:
    """Всё, что LLM получает о ходе. Полей для эталона и элементов доклада нет."""

    role_title: str
    action: SupervisorAction
    draft: str
    facts: tuple[FactLine, ...] = ()
    history: tuple[HistoryLine, ...] = ()
    last_utterance: str | None = None


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    instructions: str
    retry_note: str
    action_hints: dict[str, str]
    max_chars: int
    max_tokens: int
    history_turns: int
    max_utterance_chars: int

    def __post_init__(self) -> None:
        missing = {a.value for a in SupervisorAction} - self.action_hints.keys()
        if missing:
            raise ValueError(f"prompt template lacks action hints: {sorted(missing)}")

    @classmethod
    def load(cls, path: Path = DEFAULT_PROMPT) -> PromptTemplate:
        data = json.loads(path.read_text("utf-8"))
        return cls(
            version=data["prompt_version"],
            instructions=data["instructions"],
            retry_note=data["retry_note"],
            action_hints=dict(data["action_hints"]),
            max_chars=int(data["max_chars"]),
            max_tokens=int(data["max_tokens"]),
            history_turns=int(data["history_turns"]),
            max_utterance_chars=int(data["max_utterance_chars"]),
        )

    def as_data(self, text: str) -> str:
        """Реплика как данные: без разметки блоков и переводов строк, с ограничением длины."""
        clean = _SPACES.sub(" ", _MARKUP.sub(" ", text)).strip()
        if len(clean) > self.max_utterance_chars:
            clean = clean[: self.max_utterance_chars].rstrip() + " …"
        return clean

    def build(self, request: PhraseRequest, *, retry_reason: str | None = None) -> str:
        history = request.history[-self.history_turns :] if self.history_turns else ()
        lines = [f"{h.speaker.value}: {self.as_data(h.text)}" for h in history]
        blocks = ["<история>", *lines, "</история>"]
        if request.last_utterance is not None:
            blocks += [
                "<реплика_диспетчера>",
                self.as_data(request.last_utterance),
                "</реплика_диспетчера>",
            ]
        blocks += ["<сведения>", *(f"{f.label}: {f.value}" for f in request.facts), "</сведения>"]
        blocks += [
            f'<черновик действие="{request.action.value}">',
            request.draft,
            "</черновик>",
        ]
        if retry_reason is not None:  # только код причины, без текста отклонённого ответа
            blocks.append(self.retry_note.format(reason=retry_reason))
        return self.instructions.format(
            role_title=request.role_title,
            action=request.action.value,
            action_hint=self.action_hints[request.action.value],
            max_chars=self.max_chars,
            data="\n".join(blocks),
        )
