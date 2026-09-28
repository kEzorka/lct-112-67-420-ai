"""Генератор структурированного эталона (6.5.3): вход — сценарий и рубрика, выход —
привязка ожидаемых действий/критериев к группам рубрики со ссылками на использованные
источники (6.10). Нормативная маршрутизация не генерируется здесь: `routing.services`
подставляется из детерминированного движка правил при оценивании попытки (D-016, C-03) —
этот генератор только документирует происхождение критерия, не выбор служб.

Исправление эталона — новая версия `reference` (C-09): вызывающий передаёт `new_version`,
исходная запись не перезаписывается. Результат детерминирован при одинаковых входах —
воспроизводится из сохранённых версий сценария, рубрики и списка фрагментов.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..contracts.common import VersionRef
from ..contracts.knowledge import KnowledgeFragment
from ..contracts.rubric import Rubric
from ..contracts.scenario import CriterionRubricLink, Scenario, ScenarioReference

_RATIONALE: dict[str, str] = {
    "card.address": "Структурное поле «адрес» карточки сверяется с опубликованным фактом.",
    "card.incident_type": "Тип происшествия карточки сверяется с опубликованным фактом.",
    "card.circumstances": "Существенные обстоятельства — семантическая проверка (D-034, M5).",
    "routing.decision": "Выбранное решение ДДС сравнивается с эталонным решением сценария.",
    "routing.services": (
        "Состав служб сверяется с решением детерминированного движка правил (D-016), "
        "а не с этим эталоном — эталон лишь фиксирует применимость критерия."
    ),
    "voice.call_made": "Факт состоявшегося звонка руководителю — по журналу событий.",
    "voice.facts_transferred": "Покрытие проверяемых элементов доклада руководителю (D-036).",
    "voice.ack_received": "Доставленное подтверждение руководителя в активном вызове (C-05).",
    "time.open": "Время открытия карточки относительно норматива 30 с (C-01).",
    "time.processing": "Время обработки относительно норматива 180 с (C-01).",
    "process.sequence": "Порядок обязательных шагов учебного цикла (C-02).",
    "manual.additions": "Ручные дополнения карточки — семантическая проверка (D-034, M5).",
    "manual.grammar": "Грамотность ручного ввода — отдельное замечание (D-034), не штраф.",
}


def generate_reference(
    scenario: Scenario,
    *,
    rubric: Rubric,
    new_version: str,
    fragments_used: Sequence[KnowledgeFragment] = (),
) -> ScenarioReference:
    """Построить версию эталона с привязкой каждого критерия рубрики к обоснованию и
    источникам. `expected_decision`/`expected_fields`/`not_applicable_criteria` переносятся
    из текущего эталона сценария без изменений — их решает автор черновика (6.5.2), не этот
    генератор."""
    fragment_ids = tuple(f.fragment_id for f in fragments_used)
    links = tuple(
        CriterionRubricLink(
            criterion_id=criterion.criterion_id,
            group_id=group.group_id,
            rationale=_RATIONALE.get(
                criterion.criterion_id,
                f"Критерий «{criterion.title}» группы «{group.title}» рубрики "
                f"{rubric.rubric_version}.",
            ),
            source_fragment_ids=fragment_ids,
        )
        for group in rubric.groups
        for criterion in group.criteria
    )
    prev = scenario.reference
    return ScenarioReference(
        reference=VersionRef(name=prev.reference.name, version=new_version),
        expected_decision=prev.expected_decision,
        expected_fields=dict(prev.expected_fields),
        not_applicable_criteria=prev.not_applicable_criteria,
        criteria_links=links,
    )
