"""Адаптер: итог игры Влада → Протокол."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog import (CONDITION_DEFAULTS, CONDITIONS, GAME_VOTES, GUILT, GUILT_LABELS, PLANT_MODE, PLANT_MODES,
                      PLANT_REDUCED, ROLE_LABELS, SWITCHES, TIMELINE, TIMELINE_MONTHS)


@dataclass(frozen=True)
class Protocol:
    total_cost: float
    contributions: dict[str, float]
    conditions: dict[str, Any]
    votes: dict[str, bool | None]
    accepted: bool
    budgets: dict[str, float] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default=())

    def supports(self, role: str) -> bool:
        """Роль поддержала протокол: голос «за». «Против» и «воздержался» — не поддержка."""
        return self.votes[role] is True

    @property
    def included(self) -> list[str]:
        """Условия, которые стали обязательствами владельцев: включённые «да/нет» и режим комбината 70%."""
        return [condition for condition in CONDITIONS
                if (condition in SWITCHES and self.conditions[condition] is True)
                or (condition == PLANT_MODE and self.conditions[condition] == PLANT_REDUCED)]

    @property
    def promised_months(self) -> int:
        return int(self.conditions[TIMELINE])

    @property
    def credit(self) -> list[str]:
        """Роли, чей взнос вместе с расходами на условия больше бюджета: превышение покрыто кредитом."""
        return [role for role in ROLE_LABELS if role in self.budgets
                and self.contributions.get(role, 0) + self.spent.get(role, 0) > self.budgets[role]]


def valid(condition: str, value: Any) -> bool:
    if condition == TIMELINE:
        return isinstance(value, int) and not isinstance(value, bool) and \
            TIMELINE_MONTHS[0] <= value <= TIMELINE_MONTHS[1]
    if condition == PLANT_MODE:
        return value in PLANT_MODES
    if condition == GUILT:
        return value in GUILT_LABELS
    return isinstance(value, bool)


def read_protocol(game_result: dict[str, Any]) -> Protocol:
    """Принимает Data.data оркестратора игры Влада или тот же объект, сохранённый в JSON."""
    warnings: list[str] = []
    roles: dict[str, str] = {}  # имя игрока → роль
    budgets = {}
    for profile in game_result.get("players") or []:
        if profile.get("id") not in ROLE_LABELS:
            warnings.append(f"Неизвестная роль {profile.get('id')} у игрока «{profile.get('name')}»: "
                            "игрок не учитывается.")
            continue
        roles[profile["name"]] = profile["id"]
        if profile.get("budget") is not None:
            budgets[profile["id"]] = profile["budget"]

    def role_of(name: str) -> str | None:
        if name not in roles:
            warnings.append(f"Неизвестный игрок «{name}»: его взнос и голос не учитываются.")
        return roles.get(name)

    state = game_result["state"]
    contributions, spent = {}, {}
    for name, record in (state.get("players") or {}).items():
        role = role_of(name)
        if role and record.get("contribution"):
            contributions[role] = record["contribution"]
        if role and record.get("spent_on_levers"):
            spent[role] = record["spent_on_levers"]

    levers = state.get("levers") or {}
    conditions = {}
    for condition, default in CONDITION_DEFAULTS.items():
        value = levers.get(condition, default)
        if not valid(condition, value):
            warnings.append(f"Недопустимое значение {value!r} условия «{CONDITIONS[condition]['label']}»: "
                            "взято значение по умолчанию.")
            value = default
        conditions[condition] = value
    for condition in levers:
        if condition not in CONDITIONS:
            warnings.append(f"Неизвестное условие {condition}: в прогнозе не учитывается.")

    # Игрок без записи о голосе считается воздержавшимся.
    votes: dict[str, bool | None] = {role: None for role in ROLE_LABELS}
    for record in (game_result.get("voting_summary") or {}).get("records") or []:
        role = role_of(record.get("player"))
        if role:
            votes[role] = GAME_VOTES.get(record.get("vote"))

    return Protocol(game_result["total_cost"], contributions, conditions, votes,
                    bool(game_result.get("consensus_reached")), budgets, spent,
                    tuple(dict.fromkeys(warnings)))
