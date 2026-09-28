"""Генерация черновика сценария для преподавателя (6.5.2).

Вход — параметры от человека (тип происшествия, локация, сложность, тема, профиль ДДС) и,
опционально, утверждённые фрагменты базы знаний (6.10, `KnowledgeIndex.search()`). Выход —
`ScenarioDraftRecord` со статусом `draft`: `Scenario` (карточка, собеседник-руководитель,
проверяемые элементы доклада, скрытый эталон) плюс `GenerationMeta` со ссылками на
использованные фрагменты и версию корпуса.

Базовая реализация — детерминированный шаблон (6.5.5): факты берутся только из входных
параметров, ничего не придумывается сверх них; число пострадавших, если не задано явно,
остаётся `unknown` (инвариант 3, «неизвестное остаётся неизвестным»). Заказчик не предоставил
96 билетов (D-050) — генератор производит только явно синтетический материал
(`Provenance.SYNTHETIC`, заголовок «СИНТЕТИКА»), реальный билет он не подделывает.

Нормативная маршрутизация сюда не входит: `routing.services` подставляется из движка правил
только при оценивании попытки (D-016, C-03); черновик не публикуется без прохождения gate —
см. `scenarios/lifecycle.py`.
"""

from __future__ import annotations

from datetime import datetime

from ..contracts.card import FieldState
from ..contracts.common import VersionRef
from ..contracts.dialogue import Interlocutor, SupervisorAction
from ..contracts.events import DispatcherDecision
from ..contracts.scenario import (
    SYNTHETIC_TITLE_PREFIX,
    CheckableItem,
    InterlocutorBrief,
    Provenance,
    ReportElement,
    Scenario,
    ScenarioFact,
    ScenarioReference,
)
from ..contracts.scenario_lifecycle import (
    GenerationInputs,
    GenerationMeta,
    ScenarioDraftRecord,
    ScenarioStatus,
)
from ..knowledge.index import KnowledgeIndex
from ..supervisor.matching import tokens

SUPERVISOR_ROLE = "supervisor"
OPERATOR_112_ROLE = "operator_112"
TEMPLATE_VERSION = "draft-template-1"
DRAFT_ROUTING_CRITERION = "routing.services"

_STREET_MARKERS = {"ул", "пр", "просп", "пер", "бул", "б-р", "ш", "наб"}
_HOUSE_MARKERS = {"д", "дом"}
_LOCATION_STOPWORDS = {"г", "ул", "д", "дом", "пр", "просп", "кв", "пос", "мкр", "стр", "корп"}
_DECISION_PATTERNS: dict[DispatcherDecision, tuple[str, ...]] = {
    DispatcherDecision.RESPOND: ("реагир", "принял к исполн", "направ", "выслал"),
    DispatcherDecision.REDIRECT: ("перенаправ", "передал", "переадрес"),
    DispatcherDecision.REFUSE: ("отказ",),
}


def _stem(word: str, length: int = 6) -> str:
    return word if len(word) <= length else word[:length]


def _significant(words: list[str], *, min_len: int = 3, limit: int = 3) -> list[str]:
    return [w for w in words if len(w) >= min_len][:limit]


def _location_pattern(location: str) -> str:
    """Улица берётся из токена сразу после маркера («ул.», «пр.», ...), дом — после «д.»/«дом»,
    как в вручную составленных синтетических сценариях («ул. Тестовая, д. 12» → «тестов 12»)."""
    toks = tokens(location)
    street = next(
        (toks[i + 1] for i, t in enumerate(toks) if t in _STREET_MARKERS and i + 1 < len(toks)),
        None,
    )
    if street is None:
        street = next((t for t in toks if not t.isdigit() and t not in _LOCATION_STOPWORDS), None)
    house = next(
        (
            toks[i + 1]
            for i, t in enumerate(toks)
            if t in _HOUSE_MARKERS and i + 1 < len(toks) and toks[i + 1].isdigit()
        ),
        None,
    )
    if house is None:
        house = next((t for t in toks if t.isdigit()), None)
    parts = [p for p in (street and _stem(street), house) if p]
    return " ".join(parts) if parts else "адрес"


def _essence_pattern(incident_type: str) -> str:
    significant = _significant(tokens(incident_type))
    return " ".join(_stem(w) for w in significant) if significant else "происшестви"


def _circumstances_pattern(topic: str, incident_type: str) -> str:
    significant = _significant(tokens(topic) or tokens(incident_type))
    return " ".join(_stem(w) for w in significant) if significant else "обстоятельств"


def _report_items(
    location: str, incident_type: str, topic: str, decision: DispatcherDecision
) -> tuple[CheckableItem, ...]:
    return (
        CheckableItem(
            item_id="essence.generated",
            element=ReportElement.ESSENCE,
            patterns=(_essence_pattern(incident_type),),
        ),
        CheckableItem(
            item_id="location.generated",
            element=ReportElement.LOCATION,
            patterns=(_location_pattern(location),),
        ),
        CheckableItem(
            item_id="circumstances.generated",
            element=ReportElement.CIRCUMSTANCES,
            patterns=(_circumstances_pattern(topic, incident_type),),
        ),
        CheckableItem(
            item_id="decision.generated",
            element=ReportElement.DDS_DECISION,
            patterns=_DECISION_PATTERNS[decision],
        ),
        CheckableItem(
            item_id="request.generated",
            element=ReportElement.REQUESTED_ACTION,
            patterns=("прошу", "просим"),
        ),
    )


def generate_draft(
    *,
    name: str,
    version: str,
    incident_type: str,
    location: str,
    difficulty: str,
    topic: str,
    dds_profile: str,
    expected_decision: DispatcherDecision,
    created_at: datetime,
    diagnostic: bool = False,
    victims_state: FieldState = FieldState.UNKNOWN,
    victims_value: str | None = None,
    knowledge_index: KnowledgeIndex | None = None,
    knowledge_query: str | None = None,
    top_k: int = 3,
) -> ScenarioDraftRecord:
    """Построить черновик сценария (статус `draft`) детерминированным шаблоном (6.5.2, 6.5.5).

    `victims_state`/`victims_value` — единственный способ заявить известное число
    пострадавших; по умолчанию оно `unknown` (инвариант 3), генератор его не придумывает.
    """
    if victims_state is FieldState.KNOWN and not victims_value:
        raise ValueError("victims_state=known requires victims_value")
    if victims_state is not FieldState.KNOWN and victims_value is not None:
        raise ValueError(f"victims_state={victims_state} must not carry victims_value")

    description = f"{topic} — тип происшествия «{incident_type}», адрес: {location}."
    published_facts = (
        ScenarioFact(
            fact_id="card.address",
            topic="address",
            label="Адрес",
            state=FieldState.KNOWN,
            value=location,
        ),
        ScenarioFact(
            fact_id="card.incident_type",
            topic="incident_type",
            label="Тип происшествия",
            state=FieldState.KNOWN,
            value=incident_type,
        ),
        ScenarioFact(
            fact_id="card.description",
            topic="description",
            label="Описание",
            state=FieldState.KNOWN,
            value=description,
        ),
        ScenarioFact(
            fact_id="card.victims",
            topic="victims",
            label="Пострадавшие",
            state=victims_state,
            value=victims_value,
        ),
    )
    supervisor = InterlocutorBrief(
        interlocutor=Interlocutor(
            role_id=SUPERVISOR_ROLE,
            published_fact_ids=(),
            allowed_actions=(
                SupervisorAction.LISTEN,
                SupervisorAction.CLARIFY,
                SupervisorAction.READ_BACK,
                SupervisorAction.CONFIRM_RECEIPT,
            ),
        ),
        report_items=_report_items(location, incident_type, topic, expected_decision),
    )
    operator_112 = InterlocutorBrief(
        interlocutor=Interlocutor(
            role_id=OPERATOR_112_ROLE, allowed_actions=(SupervisorAction.LISTEN,)
        ),
        enabled=False,  # голосовой канал оператора 112 выключен (D-030)
    )
    not_applicable = (DRAFT_ROUTING_CRITERION,) if diagnostic else ()
    reference = ScenarioReference(
        reference=VersionRef(name=f"{name}.ref", version="1"),
        expected_decision=expected_decision,
        expected_fields={"address": location, "incident_type": incident_type},
        not_applicable_criteria=not_applicable,
    )
    scenario = Scenario(
        scenario=VersionRef(name=name, version=version),
        title=f"{SYNTHETIC_TITLE_PREFIX}: {incident_type} — {location}",
        provenance=Provenance.SYNTHETIC,
        incident_type=incident_type,
        dds_profile=dds_profile,
        diagnostic=diagnostic,
        published_facts=published_facts,
        interlocutors=(supervisor, operator_112),
        reference=reference,
    )

    fragment_ids: tuple[str, ...] = ()
    corpus_version = None
    if knowledge_index is not None:
        query = knowledge_query or f"{incident_type} {topic}"
        hits = knowledge_index.search(query, top_k=top_k)
        fragment_ids = tuple(h.fragment.fragment_id for h in hits)
        corpus_version = knowledge_index.corpus

    generation = GenerationMeta(
        inputs=GenerationInputs(
            incident_type=incident_type,
            location=location,
            difficulty=difficulty,
            topic=topic,
            dds_profile=dds_profile,
        ),
        template_version=TEMPLATE_VERSION,
        corpus_version=corpus_version,
        used_fragment_ids=fragment_ids,
    )
    return ScenarioDraftRecord(
        scenario=scenario,
        status=ScenarioStatus.DRAFT,
        generation=generation,
        created_at=created_at,
    )
