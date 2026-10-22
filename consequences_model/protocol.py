"""Адаптер: итог игры ядра переговоров → Протокол."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .catalog import COMPONENT_LABELS, INDEPENDENT_MONITORS, MONITORING, NGO, ROLE_LABELS, WITHOUT_MEANS


@dataclass(frozen=True)
class Protocol:
    total_cost: int
    contributions: dict[str, int]
    responsible: dict[str, str | None]
    votes: dict[str, bool | None]
    accepted: bool
    budgets: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default=())

    @property
    def monitoring_is_independent(self) -> bool:
        """Учебник, табл. 4: независимая лаборатория, а не «своя»."""
        return self.responsible[MONITORING] in INDEPENDENT_MONITORS

    def lacks_means(self, component: str) -> bool:
        """Методичка: у жителей и НКО нет денег и полномочий строить и возить воду; контроль — область НКО."""
        owner = self.responsible[component]
        return owner in WITHOUT_MEANS and not (component == MONITORING and owner == NGO)

    @property
    def credit(self) -> list[str]:
        """Роли, чей взнос больше бюджета: превышение покрыто кредитом."""
        return [role for role, amount in self.contributions.items()
                if role in self.budgets and amount > self.budgets[role]]


def read_protocol(game_result: dict[str, Any], players: list[dict[str, Any]] | None = None) -> Protocol:
    """Принимает Data.data оркестратора ({"result": ...}) или голый result.json.

    players — профили игроков с бюджетами; без них правила о кредите не срабатывают.
    """
    result = game_result["result"] if "result" in game_result else game_result
    fields = result["final_state"]["fields"]
    warnings: list[str] = []

    def value(path: str) -> Any:
        records = fields.get(path) or []
        distinct = {json.dumps(record["value"], sort_keys=True) for record in records}
        if len(distinct) > 1:
            warnings.append(f"Конфликт в поле {path}: обязательство считается незаявленным.")
            return None
        return records[0]["value"] if records else None

    contributions = {}
    for role in ROLE_LABELS:
        amount = value(f"participants.{role}.contribution")
        if amount:
            contributions[role] = amount
    responsible = {}
    for component in COMPONENT_LABELS:
        path = f"components.{component}.responsible"
        owner = value(path)
        if owner is not None and owner not in ROLE_LABELS:
            warnings.append(f"Неизвестная роль {owner} в поле {path}: компонент без ответственного.")
            owner = None
        responsible[component] = owner
    for path in fields:
        kind, ident = path.split(".")[:2]
        if kind == "components" and ident not in COMPONENT_LABELS:
            warnings.append(f"Неизвестный компонент {ident}: в прогнозе не учитывается.")
        if kind == "participants" and ident not in ROLE_LABELS:
            warnings.append(f"Неизвестная роль {ident} в поле {path}: взнос не учитывается.")
    # True — «за», False — «против», None — воздержался или не голосовал.
    raw_votes = result.get("votes") or {}
    votes = {role: raw_votes.get(role) if raw_votes.get(role) in (True, False) else None for role in ROLE_LABELS}
    budgets = {player["id"]: player["budget"] for player in players or [] if player.get("id") in ROLE_LABELS}
    return Protocol(result["total_cost"], contributions, responsible, votes,
                    bool(result.get("accepted")), budgets, tuple(warnings))
