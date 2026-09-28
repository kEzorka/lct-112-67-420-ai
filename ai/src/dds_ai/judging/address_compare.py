"""Структурное сравнение адреса (D-034, W-01): не строкой целиком, а по компонентам —
чтобы отличить неполный адрес (дом/квартира не указаны, но улица и населённый пункт верны,
0,5 по W-01) от неверного адреса (другой населённый пункт, улица или дом — критическая ошибка).

Детерминированный модуль, не LLM: адрес — структурированное поле, D-034 требует для него
«точное сравнение с разрешённой нормализацией», а не семантическую проверку.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .. import facts_guard
from ..text_norm import tokens

# Группы, определяющие место (населённый пункт/улица): расхождение или отсутствие — другой
# адрес (критическая ошибка W-01 №2), не неполный.
LOCATION_GROUPS = frozenset(
    {"city", "settlement", "street", "avenue", "lane", "highway", "boulevard", "square"}
)
# «Дом» — тоже критическая группа, но только при расхождении значения; отсутствие дома в
# ответе диспетчера — неполный адрес (0,5 по W-01), а не критическая ошибка (см. пример W-01).
HOUSE_GROUP = "house"
# Уточняющие группы: отсутствие или расхождение — неполный адрес (0,5), никогда не критическая
# ошибка (в критических ошибках W-01 названы только населённый пункт, улица и дом).
MINOR_GROUPS = frozenset({"building", "flat", "district", "entrance", "floor"})

GROUP_LABELS = {
    "city": "населённый пункт",
    "settlement": "населённый пункт",
    "street": "улица",
    "avenue": "проспект",
    "lane": "переулок",
    "highway": "шоссе",
    "boulevard": "бульвар",
    "square": "площадь",
    "house": "дом",
    "building": "корпус/строение",
    "flat": "квартира",
    "district": "микрорайон",
    "entrance": "подъезд",
    "floor": "этаж",
}


class AddressMatch(StrEnum):
    EXACT = "exact"
    INCOMPLETE = "incomplete"  # критические группы совпали, не хватает дома/уточнения
    WRONG = "wrong"  # населённый пункт, улица или дом отличаются — критическая ошибка
    UNPARSEABLE = "unparseable"  # структуру не выделить — решает эксперт, не штраф


@dataclass(frozen=True)
class AddressComparison:
    match: AddressMatch
    differing_groups: tuple[str, ...] = ()
    missing_groups: tuple[str, ...] = ()

    def labels(self, groups: tuple[str, ...]) -> str:
        return ", ".join(GROUP_LABELS.get(g, g) for g in groups)


def _group_values(address: str) -> dict[str, list[str]]:
    """Группа → значение (1–2 токена сразу после маркера группы)."""
    toks = tokens(address)
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(toks):
        group = facts_guard.ADDRESS_GROUPS.get(toks[i])
        if group is None:
            i += 1
            continue
        j = i + 1
        value: list[str] = []
        while j < len(toks) and toks[j] not in facts_guard.ADDRESS_GROUPS and len(value) < 2:
            value.append(toks[j])
            j += 1
        if value:
            out.setdefault(group, []).extend(value)
        i = j if j > i + 1 else i + 1
    return out


def compare(actual: str, expected: str) -> AddressComparison:
    """Сравнить адрес диспетчера с эталонным. Не сама строка — группы компонентов."""
    a, e = _group_values(actual), _group_values(expected)
    if not a or not e:
        return AddressComparison(AddressMatch.UNPARSEABLE)

    wrong = sorted(g for g in LOCATION_GROUPS if g in e and a.get(g) != e.get(g))
    if HOUSE_GROUP in e and HOUSE_GROUP in a and a[HOUSE_GROUP] != e[HOUSE_GROUP]:
        wrong.append(HOUSE_GROUP)
    if wrong:
        return AddressComparison(AddressMatch.WRONG, differing_groups=tuple(sorted(wrong)))

    incomplete = sorted(
        g for g in ({HOUSE_GROUP} | MINOR_GROUPS) if g in e and a.get(g) != e.get(g)
    )
    if incomplete:
        return AddressComparison(AddressMatch.INCOMPLETE, missing_groups=tuple(incomplete))
    return AddressComparison(AddressMatch.EXACT)
