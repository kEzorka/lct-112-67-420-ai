"""Мок детерминированного движка маршрутизации: таблица правил, без моделей."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.common import VersionRef
from ..contracts.criteria import RuleStatus
from ..contracts.routing import RoutingDecision, RoutingRequest


@dataclass(frozen=True)
class Rule:
    rule_id: str
    status: RuleStatus
    ekp_code: str | None
    main_service: str | None
    services: tuple[str, ...]


class TableRoutingEngine:
    def __init__(self, rules: dict[str, Rule], version: str = "mock-1"):
        self._rules = dict(rules)
        self.version = VersionRef(name="routing_rules", version=version)

    def route(self, request: RoutingRequest) -> RoutingDecision:
        rule = self._rules.get(request.incident_type)
        if rule is None:
            return RoutingDecision(routing_rules=self.version, rule_id=None, rule_status=None)
        return RoutingDecision(
            routing_rules=self.version,
            rule_id=rule.rule_id,
            rule_status=rule.status,
            ekp_code=rule.ekp_code,
            main_service=rule.main_service,
            services=rule.services,
        )
