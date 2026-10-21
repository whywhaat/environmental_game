"""Прогноз по итогу игры: протокол → исполнение → состояние города → исход."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .catalog import (ADMIN, CITY_PAID_ALL_SHARE, DEFAULT_PROMISED_MONTHS, DELAY_PARTIAL_RECONSTRUCTION,
                      DELAY_UNDERFUNDED, FULL, HOLDING, HOLDING_PAID_ALL_SHARE, HORIZON_MONTHS, MIN_COVERAGE_TO_BUILD,
                      MONITORING, NGO, NONE, OUTCOMES, PAID_PART, PARTIAL, RAISE, RECONSTRUCTION, RESIDENTS,
                      ROLE_LABELS, WATER)
from .explain import fired_rules
from .protocol import Protocol, read_protocol
from .report import render_herald_prompt, render_report
from .state import city_state, fates


def execute(protocol: Protocol, excluded: str | None = None) -> dict[str, Any]:
    """Шаг 1: насколько каждая роль выполнила свои обязательства.

    excluded — роль, чьи обязательства считаются невыполненными (контрфактический прогон).
    """
    def by_vote(owner: str | None) -> str:
        if owner is None or owner == excluded:
            return NONE
        if protocol.accepted:
            return FULL if protocol.votes[owner] else PARTIAL
        return PARTIAL if protocol.votes[owner] else NONE

    contributions = {role: by_vote(role) for role in protocol.contributions}
    components = {component: by_vote(owner) for component, owner in protocol.responsible.items()}
    if protocol.accepted and components[MONITORING] == FULL and protocol.monitoring_is_independent:
        # Учебник, табл. 5, «Право»: срыв виден, и его исправляют.
        contributions = {role: value if role == excluded else RAISE[value] for role, value in contributions.items()}
        components = {component: value if component == MONITORING or protocol.responsible[component] in (None, excluded)
                      else RAISE[value] for component, value in components.items()}
    capped = [component for component in components
              if protocol.lacks_means(component) and components[component] == FULL]
    for component in capped:
        components[component] = PARTIAL
    return {"contributions": contributions, "components": components, "capped": capped}


def paid_in(protocol: Protocol, execution: dict[str, Any]) -> float:
    """Сколько денег реально внесено: полностью — весь взнос, частично — половина."""
    return sum(amount * PAID_PART[execution["contributions"][role]] for role, amount in protocol.contributions.items())


def commissioning(protocol: Protocol, execution: dict[str, Any],
                  promised_months: int) -> tuple[int | None, list[tuple[int, str]]]:
    """Месяц ввода очистных от голосования (None — не введены к году 10) и задержки с причинами."""
    coverage = paid_in(protocol, execution) / protocol.total_cost
    reconstruction = execution["components"][RECONSTRUCTION]
    if reconstruction == NONE or coverage < MIN_COVERAGE_TO_BUILD:
        return None, []
    delays = []
    if reconstruction == PARTIAL:
        delays.append((DELAY_PARTIAL_RECONSTRUCTION, "реконструкция выполнена частично"))
    if coverage < 1:
        delays.append((DELAY_UNDERFUNDED, "смета покрыта не полностью"))
    month = promised_months + sum(months for months, _ in delays)
    return (month, delays) if month <= HORIZON_MONTHS else (None, delays)


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
        and all(value == FULL for value in f["components"].values())
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
    month, delays = commissioning(protocol, execution, promised_months)
    facts = {
        "month": month,
        "delays": delays,
        "promised_months": promised_months,
        "paid": paid_in(protocol, execution),
        "contributions": execution["contributions"],
        "components": execution["components"],
        "capped": execution["capped"],
        "levels": list(execution["contributions"].values()) + list(execution["components"].values()),
        "responsible": protocol.responsible,
        "votes": protocol.votes,
        "shares": shares(protocol),
        "credit": protocol.credit,
        "monitoring_ok": execution["components"][MONITORING] == FULL and protocol.monitoring_is_independent,
    }
    return {"execution": execution, "facts": facts, "outcome": choose_outcome(protocol, facts)}


def forecast(game_result: dict[str, Any], players: list[dict[str, Any]] | None = None,
             promised_months: int = DEFAULT_PROMISED_MONTHS) -> dict[str, Any]:
    """Прогноз по итогу игры; players — профили с бюджетами, promised_months — обещанный срок ввода."""
    protocol = read_protocol(game_result, players)
    run = simulate(protocol, promised_months)
    facts = run["facts"]
    state = city_state(facts)
    without_role = {role: simulate(protocol, promised_months, excluded=role)["outcome"]["name"]
                    for role in ROLE_LABELS}
    result = {
        "outcome": run["outcome"],
        "protocol": {**asdict(protocol), "credit": protocol.credit},
        "promised_months": promised_months,
        "paid": facts["paid"],
        "execution": {key: run["execution"][key] for key in ("contributions", "components")},
        "commissioning_month": facts["month"],
        "state": state,
        "fates": fates(facts, state),
        "explanation": {"rules": fired_rules(protocol, facts, state, run["outcome"]), "without_role": without_role},
        "warnings": list(protocol.warnings),
    }
    result["report"] = render_report(result)
    result["herald_prompt"] = render_herald_prompt(result)
    return result
