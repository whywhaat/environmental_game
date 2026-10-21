"""Прогноз по итогу игры: протокол → исполнение → состояние города → исход."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .catalog import (ADMIN, CITY_PAID_ALL_SHARE, DEFAULT_PROMISED_MONTHS, DELAY_PARTIAL_RECONSTRUCTION,
                      DELAY_UNDERFUNDED, FULL, HOLDING, HOLDING_PAID_ALL_SHARE, HORIZON_MONTHS, INDEPENDENT_MONITORS,
                      MIN_COVERAGE_TO_BUILD, MONITORING, NGO, NONE, OUTCOMES, PAID_PART, PARTIAL, RAISE,
                      RECONSTRUCTION, RESIDENTS, ROLE_LABELS, WATER, WITHOUT_MEANS)
from .explain import fired_rules
from .protocol import Protocol, read_protocol
from .report import render_herald_prompt, render_report
from .state import city_state, fates


def execute(protocol: Protocol, excluded: str | None = None) -> dict[str, dict[str, str]]:
    """Шаг 1: насколько каждая роль выполнила свои обязательства.

    excluded — роль, чьи обязательства считаются невыполненными (контрфактический прогон).
    """
    def level(owner: str | None) -> str:
        if owner is None or owner == excluded:
            return NONE
        if protocol.accepted:
            return FULL if protocol.votes[owner] else PARTIAL
        return PARTIAL if protocol.votes[owner] else NONE

    contributions = {role: level(role) for role in protocol.contributions}
    components = {component: level(owner) for component, owner in protocol.responsible.items()}
    if protocol.accepted and components[MONITORING] == FULL and is_independent(protocol):
        # Учебник, табл. 5, «Право»: срыв виден, и его исправляют.
        contributions = {role: value if role == excluded else RAISE[value] for role, value in contributions.items()}
        components = {component: RAISE[value] if component != MONITORING and owner not in (None, excluded)
                      else value
                      for (component, value), owner in zip(components.items(), protocol.responsible.values())}
    for component, owner in protocol.responsible.items():
        # Методичка: у жителей и НКО нет денег и полномочий строить и возить воду.
        if owner in WITHOUT_MEANS and not (component == MONITORING and owner == NGO) and components[component] == FULL:
            components[component] = PARTIAL
    return {"contributions": contributions, "components": components}


def is_independent(protocol: Protocol) -> bool:
    """Учебник, табл. 4: независимая лаборатория, а не «своя»."""
    return protocol.responsible[MONITORING] in INDEPENDENT_MONITORS


def paid_in(protocol: Protocol, execution: dict[str, dict[str, str]]) -> float:
    """Сколько денег реально внесено: полностью — весь взнос, частично — половина."""
    return sum(amount * PAID_PART[execution["contributions"][role]] for role, amount in protocol.contributions.items())


def commissioning_month(protocol: Protocol, execution: dict[str, dict[str, str]],
                        promised_months: int) -> int | None:
    """Месяц ввода очистных от голосования; None — не введены к году 10."""
    coverage = paid_in(protocol, execution) / protocol.total_cost
    reconstruction = execution["components"][RECONSTRUCTION]
    if reconstruction == NONE or coverage < MIN_COVERAGE_TO_BUILD:
        return None
    month = promised_months
    if reconstruction == PARTIAL:
        month += DELAY_PARTIAL_RECONSTRUCTION
    if coverage < 1:
        month += DELAY_UNDERFUNDED
    return month if month <= HORIZON_MONTHS else None


def shares(protocol: Protocol) -> dict[str, float]:
    """Доля каждой роли в смете по заявленным взносам."""
    return {role: protocol.contributions.get(role, 0) / protocol.total_cost for role in ROLE_LABELS}


# Исходы проверяются по порядку, срабатывает первый подходящий.
ACCEPTED_RULES = (
    ("nedostroy", lambda f: f["month"] is None),
    ("without_residents", lambda f: not f["votes"][RESIDENTS] and not f["votes"][NGO]),
    ("dolgostroy", lambda f: f["month"] > f["promised_months"] and f["components"][WATER] != FULL),
    ("holding_paid", lambda f: f["shares"][HOLDING] >= HOLDING_PAID_ALL_SHARE),
    ("city_paid", lambda f: f["shares"][ADMIN] >= CITY_PAID_ALL_SHARE),
    ("open_river", lambda f: f["month"] == f["promised_months"] and f["monitoring_ok"]
        and all(level == FULL for level in f["components"].values())
        and f["votes"][RESIDENTS] and f["votes"][NGO]),
    ("closed_data", lambda f: not f["monitoring_ok"]),
    ("compromise", lambda f: True),
)
REJECTED_RULES = (
    ("each_for_self", lambda f: PARTIAL in f["levels"]),
    ("stagnation", lambda f: True),
)


def choose_outcome(protocol: Protocol, facts: dict[str, Any]) -> dict[str, str]:
    rules = ACCEPTED_RULES if protocol.accepted else REJECTED_RULES
    outcome_id = next(outcome_id for outcome_id, applies in rules if applies(facts))
    return {"id": outcome_id, **OUTCOMES[outcome_id]}


def simulate(protocol: Protocol, promised_months: int, excluded: str | None = None) -> dict[str, Any]:
    execution = execute(protocol, excluded)
    month = commissioning_month(protocol, execution, promised_months)
    facts = {
        "month": month,
        "promised_months": promised_months,
        "paid": paid_in(protocol, execution),
        "contributions": execution["contributions"],
        "components": execution["components"],
        "levels": list(execution["contributions"].values()) + list(execution["components"].values()),
        "responsible": protocol.responsible,
        "votes": protocol.votes,
        "shares": shares(protocol),
        "credit": protocol.credit,
        "monitoring_ok": execution["components"][MONITORING] == FULL and is_independent(protocol),
    }
    return {"execution": execution, "facts": facts, "outcome": choose_outcome(protocol, facts)}


def forecast(game_result: dict[str, Any], players: list[dict[str, Any]] | None = None,
             promised_months: int = DEFAULT_PROMISED_MONTHS) -> dict[str, Any]:
    """Прогноз по итогу игры; players — профили с бюджетами, promised_months — обещанный срок ввода."""
    protocol = read_protocol(game_result, players)
    run = simulate(protocol, promised_months)
    facts = run["facts"]
    state = city_state(facts, HORIZON_MONTHS)
    without_role = {role: simulate(protocol, promised_months, excluded=role)["outcome"]["name"]
                    for role in ROLE_LABELS}
    result = {
        "outcome": run["outcome"],
        "protocol": {**asdict(protocol), "credit": protocol.credit},
        "promised_months": promised_months,
        "paid": facts["paid"],
        "execution": run["execution"],
        "commissioning_month": facts["month"],
        "state": state,
        "fates": fates(facts, state),
        "explanation": {"rules": fired_rules(protocol, facts, run["outcome"]), "without_role": without_role},
        "warnings": list(protocol.warnings),
    }
    result["report"] = render_report(result)
    result["herald_prompt"] = render_herald_prompt(result)
    return result
