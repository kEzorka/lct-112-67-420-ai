"""Сценарий — минимальный черновик контракта для ИИ-руководителя (6.4, 6.5, D-003, D-030).

Разделение по видимости:
- `published_facts` — факты, опубликованные во входной карточке (видит обучаемый);
- `interlocutors[*].facts` — факты собеседника (руководитель может их назвать);
- `interlocutors[*].report_items` — проверяемые элементы доклада: руководитель по ним
  сопоставляет доклад, но в реплики их не подставляет;
- `reference` — скрытый эталон оценивания: руководителю не передаётся вообще (инвариант 1).

Полный формат сценария (вопросы, сложность, ссылки на источники) — M4.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .card import FieldState
from .common import Contract, NonEmptyStr, VersionRef
from .dialogue import Interlocutor, ScenarioCallState
from .events import DispatcherDecision


class Provenance(StrEnum):
    """Синтетический материал не называется реальным билетом (6.5)."""

    SYNTHETIC = "synthetic"
    TICKET = "ticket"


SYNTHETIC_TITLE_PREFIX = "СИНТЕТИКА"


class ScenarioFact(Contract):
    """Факт сценария. «Неизвестно», «нет», «не применимо» — состояния, а не пустые значения."""

    fact_id: NonEmptyStr
    topic: NonEmptyStr = Field(description="Имя поля карточки или тема факта: address, victims")
    label: NonEmptyStr = Field(description="Как факт называется в реплике: «Число пострадавших»")
    state: FieldState
    value: str | None = None
    variant_values: tuple[NonEmptyStr, ...] = Field(
        default=(),
        description=(
            "Разрешённые равнозначные формулировки value для вариации карточки (6.5): "
            "автор сценария заранее утверждает их как факт, генератор не придумывает новые"
        ),
    )
    omissible: bool = Field(
        default=False,
        description="Известный факт может быть опущен в сгенерированной карточке (6.5)",
    )
    question_patterns: tuple[NonEmptyStr, ...] = Field(
        default=(),
        description="Шаблоны вопроса о факте: основы слов через пробел, все должны встретиться",
    )

    @model_validator(mode="after")
    def _state(self) -> ScenarioFact:
        if self.state is FieldState.KNOWN and not self.value:
            raise ValueError("known fact requires value")
        if self.state is not FieldState.KNOWN and self.value is not None:
            raise ValueError(f"{self.state} fact must not carry a value")
        if self.state is not FieldState.KNOWN and self.variant_values:
            raise ValueError(f"{self.state} fact must not carry variant_values")
        return self


class ReportElement(StrEnum):
    """Проверяемые элементы доклада руководителю (6.4)."""

    ESSENCE = "essence"
    LOCATION = "location"
    CIRCUMSTANCES = "circumstances"
    DDS_DECISION = "dds_decision"
    REQUESTED_ACTION = "requested_action"


class CheckableItem(Contract):
    """Проверяемый элемент доклада. На M1 сопоставляется по ключевым словам.

    Шаблон — основы слов через пробел; элемент найден, если в реплике есть все основы хотя бы
    одного шаблона. Числа и основы с `$` на конце («пожар$») сравниваются как слово целиком.
    """

    item_id: NonEmptyStr
    element: ReportElement
    required: bool = True
    patterns: tuple[NonEmptyStr, ...] = Field(min_length=1)


class InterlocutorBrief(Contract):
    """Всё, что известно собеседнику (D-030): роль, его факты, элементы доклада, сценарий вызова.

    Эталон оценивания сюда не входит.
    """

    interlocutor: Interlocutor
    facts: tuple[ScenarioFact, ...] = ()
    report_items: tuple[CheckableItem, ...] = ()
    dial_outcomes: tuple[ScenarioCallState, ...] = Field(
        default=(),
        description="Сценарные исходы наборов номера до соединения: busy / no_answer",
    )
    drop_after_turns: int | None = Field(
        default=None, ge=1, description="Сценарный обрыв первого соединения после N реплик"
    )
    enabled: bool = Field(default=True, description="Голосовой канал оператора 112 выключен")

    @model_validator(mode="after")
    def _consistent(self) -> InterlocutorBrief:
        ids = [f.fact_id for f in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact_id must be unique within a brief")
        if set(ids) != set(self.interlocutor.published_fact_ids):
            raise ValueError("brief facts must equal interlocutor.published_fact_ids")
        if ScenarioCallState.DROPPED in self.dial_outcomes:
            raise ValueError("dropped is set by drop_after_turns, not by a dial outcome")
        items = [i.item_id for i in self.report_items]
        if len(items) != len(set(items)):
            raise ValueError("item_id must be unique within a brief")
        if self.report_items:
            missing = set(ReportElement) - {i.element for i in self.report_items}
            if missing:
                raise ValueError(f"report items do not cover elements: {sorted(missing)}")
        return self

    @property
    def role_id(self) -> str:
        return self.interlocutor.role_id


class CriterionRubricLink(Contract):
    """Связь ожидаемого результата с критерием рубрики и источником (6.5, генератор эталона)."""

    criterion_id: NonEmptyStr
    group_id: NonEmptyStr
    rationale: NonEmptyStr = Field(description="Понятное обоснование для критерия по эталону")
    source_fragment_ids: tuple[NonEmptyStr, ...] = Field(
        default=(), description="Фрагменты базы знаний (6.10), использованные при генерации"
    )


class ScenarioReference(Contract):
    """Скрытый эталон: не виден обучаемому и не передаётся собеседникам (инвариант 1)."""

    reference: VersionRef
    expected_decision: DispatcherDecision
    expected_fields: dict[NonEmptyStr, NonEmptyStr] = Field(
        default_factory=dict,
        description="Ожидаемые значения структурированных полей карточки после дополнения",
    )
    not_applicable_criteria: tuple[NonEmptyStr, ...] = Field(
        default=(), description="Неприменимость критериев задаётся до старта (C-06)"
    )
    criteria_links: tuple[CriterionRubricLink, ...] = Field(
        default=(),
        description="Привязка ожидаемых действий/ответов к группам рубрики и источникам (6.5)",
    )


class Scenario(Contract):
    scenario: VersionRef
    title: NonEmptyStr
    provenance: Provenance
    source_ref: str | None = Field(
        default=None, description="Исходный билет; у производных вариантов — ссылка на него"
    )
    incident_type: NonEmptyStr = Field(description="Вход движка маршрутизации, не выбор служб")
    dds_profile: NonEmptyStr = Field(
        default="profile-1", description="Условный профиль ДДС пилота (D-019, W-03)"
    )
    diagnostic: bool = Field(
        default=False,
        description=(
            "Диагностический сценарий на выявление неопределённости маршрута (C-03): "
            "не даёт автоматического штрафа/балла за маршрут и не публикуется как обычный"
        ),
    )
    published_facts: tuple[ScenarioFact, ...] = Field(min_length=1)
    interlocutors: tuple[InterlocutorBrief, ...] = Field(min_length=1)
    reference: ScenarioReference

    @model_validator(mode="after")
    def _consistent(self) -> Scenario:
        if self.provenance is Provenance.SYNTHETIC:
            if not self.title.startswith(SYNTHETIC_TITLE_PREFIX):
                raise ValueError(
                    f"synthetic scenario title must start with {SYNTHETIC_TITLE_PREFIX}"
                )
            if self.source_ref is not None:
                raise ValueError("synthetic scenario must not reference a ticket")
        elif not self.source_ref:
            raise ValueError("ticket scenario requires source_ref")
        roles = [b.role_id for b in self.interlocutors]
        if len(roles) != len(set(roles)):
            raise ValueError("role_id must be unique")
        topics = [f.topic for f in self.published_facts]
        if len(topics) != len(set(topics)):
            raise ValueError("published fact topics must be unique (one card field each)")
        known_topics = {f.topic for f in self.published_facts if f.state is FieldState.KNOWN}
        unknown_expected = set(self.reference.expected_fields) - known_topics
        if unknown_expected:
            raise ValueError(
                f"expected_fields reference topics without a known published fact: "
                f"{sorted(unknown_expected)}"
            )
        return self

    def brief(self, role_id: str) -> InterlocutorBrief:
        for b in self.interlocutors:
            if b.role_id == role_id:
                return b
        raise KeyError(role_id)
