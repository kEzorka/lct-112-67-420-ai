"""Выход детерминированного движка маршрутизации (D-016, C-03).

ИИ-контур только читает RoutingDecision; службы никогда не выбирает LLM.
"""

from __future__ import annotations

from pydantic import Field

from .common import Contract, NonEmptyStr, VersionRef
from .criteria import RuleStatus


class RoutingRequest(Contract):
    incident_type: NonEmptyStr
    features: dict[str, str] = Field(default_factory=dict)
    address_known: bool = True


class RoutingDecision(Contract):
    routing_rules: VersionRef
    rule_id: str | None
    rule_status: RuleStatus | None = Field(description="None — правило не найдено")
    ekp_code: str | None = None
    main_service: str | None = None
    services: tuple[NonEmptyStr, ...] = ()

    @property
    def auto_gradable(self) -> bool:
        return self.rule_status is RuleStatus.ACTIVE and self.main_service is not None
