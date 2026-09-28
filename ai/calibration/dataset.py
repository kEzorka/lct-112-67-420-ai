"""Контрольная выборка для калибровки семантического оценивателя (D-039).

Разметка `expected` — рабочая метка команды для отладки инструмента, **не экспертная
разметка преподавателя** (это явно отражено в отчёте `docs/ai/calibration-report.md`, не
только здесь). Кейсы синтетические: сгенерированы вручную командой для покрытия классов
раздела 8 промпта — правильные, частично правильные, ошибочные, перефразирования,
искажённые транскрипты. Исходный факт и его производные (перефразирование/ошибка/порча)
делят один `source_ref`, чтобы при любом разбиении выборки они не оказались в разных частях
(требование D-039 «исходный билет и все его производные — в одной выборке»).

Эти кейсы не использовались для настройки текста промпта (`ai/config/judge.prompt.ru.json`):
промпт написан один раз до прогона калибровки и не правился по её результатам — так и
остаётся, чтобы контрольная выборка не превратилась в обучающую.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    CORRECT = "correct"
    PARAPHRASE = "paraphrase"
    PARTIAL = "partial"
    WRONG = "wrong"
    CORRUPTED = "corrupted_stt"


class ExpectedLabel(StrEnum):
    """Совпадает по значениям с `judging.schema.MatchLevel` — рабочая метка команды."""

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class FieldCase:
    case_id: str
    source_ref: str  # группировка исходник+производные (D-039)
    category: Category
    criterion_id: str
    expected_label: ExpectedLabel
    expected_fact_label: str
    expected_fact_values: tuple[str, ...]
    student_text: str
    note: str = ""

    @property
    def context(self) -> dict:
        return {
            "kind": "field_fact",
            "field_name": self.criterion_id,
            "student_text": self.student_text,
            "expected_facts": [
                {"label": self.expected_fact_label, "values": list(self.expected_fact_values)}
            ],
        }


@dataclass(frozen=True)
class ConversationCase:
    case_id: str
    source_ref: str
    category: Category
    criterion_id: str
    expected_label: ExpectedLabel
    item_id: str
    item_description: str
    transcript_text: str
    note: str = ""

    @property
    def context(self) -> dict:
        return {
            "kind": "conversation_coverage",
            "transcript": [{"event_id": self.case_id, "text": self.transcript_text}],
            "items": [{"item_id": self.item_id, "description": self.item_description}],
        }


FIELD_CASES: tuple[FieldCase, ...] = (
    # --- source: smoke-from-window (card.circumstances) ---
    FieldCase(
        "F-smoke-01",
        "smoke-from-window",
        Category.CORRECT,
        "card.circumstances",
        ExpectedLabel.FULL,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "Дым идёт из окна квартиры соседей.",
    ),
    FieldCase(
        "F-smoke-02",
        "smoke-from-window",
        Category.PARAPHRASE,
        "card.circumstances",
        ExpectedLabel.FULL,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "Наблюдается задымление из окна соседней квартиры.",
    ),
    FieldCase(
        "F-smoke-03",
        "smoke-from-window",
        Category.PARAPHRASE,
        "card.circumstances",
        ExpectedLabel.FULL,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "Жильцы видят дым, который идёт из окна квартиры по соседству.",
    ),
    FieldCase(
        "F-smoke-04",
        "smoke-from-window",
        Category.PARTIAL,
        "card.circumstances",
        ExpectedLabel.PARTIAL,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "Что-то странное происходит у соседей, слышен шум.",
        "упомянуто беспокойство у соседей, но не дым и не окно",
    ),
    FieldCase(
        "F-smoke-05",
        "smoke-from-window",
        Category.WRONG,
        "card.circumstances",
        ExpectedLabel.NONE,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "Пожара нет, всё спокойно, ложный вызов.",
    ),
    FieldCase(
        "F-smoke-06",
        "smoke-from-window",
        Category.CORRUPTED,
        "card.circumstances",
        ExpectedLabel.UNCLEAR,
        "Обстоятельства",
        ("дым идёт из окна квартиры соседей",),
        "ды... ##помех## из ок... кварт... соседе...",
    ),
    # --- source: nine-floor-house (manual.additions) ---
    FieldCase(
        "F-floors-01",
        "nine-floor-house",
        Category.CORRECT,
        "manual.additions",
        ExpectedLabel.FULL,
        "Этажность дома",
        ("9 этажей", "девять этажей"),
        "Дом девятиэтажный.",
    ),
    FieldCase(
        "F-floors-02",
        "nine-floor-house",
        Category.PARAPHRASE,
        "manual.additions",
        ExpectedLabel.FULL,
        "Этажность дома",
        ("9 этажей", "девять этажей"),
        "В доме девять этажей.",
    ),
    FieldCase(
        "F-floors-03",
        "nine-floor-house",
        Category.PARTIAL,
        "manual.additions",
        ExpectedLabel.PARTIAL,
        "Этажность дома",
        ("9 этажей", "девять этажей"),
        "Дом многоэтажный.",
        "многоэтажный не уточняет число этажей",
    ),
    FieldCase(
        "F-floors-04",
        "nine-floor-house",
        Category.WRONG,
        "manual.additions",
        ExpectedLabel.NONE,
        "Этажность дома",
        ("9 этажей", "девять этажей"),
        "Дом пятиэтажный.",
    ),
    FieldCase(
        "F-floors-05",
        "nine-floor-house",
        Category.CORRUPTED,
        "manual.additions",
        ExpectedLabel.UNCLEAR,
        "Этажность дома",
        ("9 этажей", "девять этажей"),
        "дом ...этаж... ##обрыв связи##",
    ),
    # --- source: gas-smell-entrance (card.circumstances, второй сценарий) ---
    FieldCase(
        "F-gas-01",
        "gas-smell-entrance",
        Category.CORRECT,
        "card.circumstances",
        ExpectedLabel.FULL,
        "Обстоятельства",
        ("запах газа в подъезде",),
        "Запах газа в подъезде.",
    ),
    FieldCase(
        "F-gas-02",
        "gas-smell-entrance",
        Category.PARAPHRASE,
        "card.circumstances",
        ExpectedLabel.FULL,
        "Обстоятельства",
        ("запах газа в подъезде",),
        "В подъезде сильно пахнет газом.",
    ),
    FieldCase(
        "F-gas-03",
        "gas-smell-entrance",
        Category.WRONG,
        "card.circumstances",
        ExpectedLabel.NONE,
        "Обстоятельства",
        ("запах газа в подъезде",),
        "В квартире протекает кран, вода на полу.",
    ),
    FieldCase(
        "F-gas-04",
        "gas-smell-entrance",
        Category.CORRUPTED,
        "card.circumstances",
        ExpectedLabel.UNCLEAR,
        "Обстоятельства",
        ("запах газа в подъезде",),
        "##шум## запа... газ... подъе...",
    ),
)

CONVERSATION_CASES: tuple[ConversationCase, ...] = (
    # --- source: report-essence-fire ---
    ConversationCase(
        "C-essence-01",
        "report-essence-fire",
        Category.CORRECT,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "essence.fire",
        "пожар / возгорание / задымление",
        "Пожар в жилом доме, дым идёт из окна квартиры.",
    ),
    ConversationCase(
        "C-essence-02",
        "report-essence-fire",
        Category.PARAPHRASE,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "essence.fire",
        "пожар / возгорание / задымление",
        "Наблюдается возгорание, в подъезде сильное задымление.",
    ),
    ConversationCase(
        "C-essence-03",
        "report-essence-fire",
        Category.WRONG,
        "voice.facts_transferred",
        ExpectedLabel.NONE,
        "essence.fire",
        "пожар / возгорание / задымление",
        "Соседи шумят, просьба отреагировать.",
    ),
    ConversationCase(
        "C-essence-04",
        "report-essence-fire",
        Category.CORRUPTED,
        "voice.facts_transferred",
        ExpectedLabel.UNCLEAR,
        "essence.fire",
        "пожар / возгорание / задымление",
        "###сигнал прерывается### пож... ##помехи## дом...",
    ),
    # --- source: report-location-address ---
    ConversationCase(
        "C-location-01",
        "report-location-address",
        Category.CORRECT,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "location.address",
        "адрес: улица Тестовая, дом 12",
        "Адрес: улица Тестовая, дом 12.",
    ),
    ConversationCase(
        "C-location-02",
        "report-location-address",
        Category.PARAPHRASE,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "location.address",
        "адрес: улица Тестовая, дом 12",
        "Происшествие по адресу: Тестовая улица, дом двенадцать.",
    ),
    ConversationCase(
        "C-location-03",
        "report-location-address",
        Category.PARTIAL,
        "voice.facts_transferred",
        ExpectedLabel.PARTIAL,
        "location.address",
        "адрес: улица Тестовая, дом 12",
        "Где-то на Тестовой улице.",
        "улица названа, номер дома — нет",
    ),
    ConversationCase(
        "C-location-04",
        "report-location-address",
        Category.WRONG,
        "voice.facts_transferred",
        ExpectedLabel.NONE,
        "location.address",
        "адрес: улица Тестовая, дом 12",
        "Пока уточняю детали происшествия.",
    ),
    ConversationCase(
        "C-location-05",
        "report-location-address",
        Category.CORRUPTED,
        "voice.facts_transferred",
        ExpectedLabel.UNCLEAR,
        "location.address",
        "адрес: улица Тестовая, дом 12",
        "адрес ...тест... ##обрыв## дом...",
    ),
    # --- source: report-decision-respond ---
    ConversationCase(
        "C-decision-01",
        "report-decision-respond",
        Category.CORRECT,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "decision.respond",
        "ДДС реагирует, направлена пожарная охрана",
        "ДДС реагирует, направляем пожарную охрану.",
    ),
    ConversationCase(
        "C-decision-02",
        "report-decision-respond",
        Category.PARAPHRASE,
        "voice.facts_transferred",
        ExpectedLabel.FULL,
        "decision.respond",
        "ДДС реагирует, направлена пожарная охрана",
        "Принимаем к исполнению, выслали пожарный расчёт.",
    ),
    ConversationCase(
        "C-decision-03",
        "report-decision-respond",
        Category.WRONG,
        "voice.facts_transferred",
        ExpectedLabel.NONE,
        "decision.respond",
        "ДДС реагирует, направлена пожарная охрана",
        "Пока не знаю, что делать с этим вызовом.",
    ),
)


def all_field_case_ids() -> tuple[str, ...]:
    return tuple(c.case_id for c in FIELD_CASES)


def all_conversation_case_ids() -> tuple[str, ...]:
    return tuple(c.case_id for c in CONVERSATION_CASES)
