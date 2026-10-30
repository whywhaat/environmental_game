"""Прогноз по итогу игры: протокол → исполнение → состояние города → исход."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .catalog import (ADMIN, CITY_PAID_ALL_SHARE, CONDITIONS, DELAY_PARTIAL_RECONSTRUCTION, DELAY_UNDERFUNDED, FULL,
                      GUILT, HOLDING, HOLDING_PAID_ALL_SHARE, HORIZON_MONTHS, MIN_COVERAGE_TO_BUILD, NGO, NONE,
                      OUTCOMES, PAID_PART, PARTIAL, RAISE, RECONSTRUCTION_OWNER, RESIDENTS, ROLE_LABELS, SENSORS,
                      WATER)
from .explain import fired_rules
from .protocol import Protocol, read_protocol
from .report import render_herald_prompt, render_report
from .state import city_state, fates


def execute(protocol: Protocol, excluded: str | None = None) -> dict[str, Any]:
    """Шаг 1: насколько каждая роль выполнила свои обязательства.

    excluded — роль, чьи обязательства считаются невыполненными (контрфактический прогон).
    """
    def by_vote(owner: str) -> str:
        if owner == excluded:
            return NONE
        if protocol.accepted:
            return FULL if protocol.supports(owner) else PARTIAL
        return PARTIAL if protocol.supports(owner) else NONE

    contributions = {role: by_vote(role) for role in protocol.contributions}
    conditions = {condition: by_vote(CONDITIONS[condition]["owner"]) for condition in protocol.included}
    reconstruction = by_vote(RECONSTRUCTION_OWNER)
    if protocol.accepted and conditions.get(SENSORS) == FULL:
        # Учебник, табл. 5, «Право»: срыв виден, и его исправляют.
        def raised(owner: str, value: str) -> str:
            return value if owner == excluded else RAISE[value]

        contributions = {role: raised(role, value) for role, value in contributions.items()}
        conditions = {condition: value if condition == SENSORS else raised(CONDITIONS[condition]["owner"], value)
                      for condition, value in conditions.items()}
        reconstruction = raised(RECONSTRUCTION_OWNER, reconstruction)
    return {"contributions": contributions, "conditions": conditions, "reconstruction": reconstruction}


def paid_in(protocol: Protocol, execution: dict[str, Any]) -> float:
    """Сколько денег реально внесено: полностью — весь взнос, частично — половина."""
    return sum(amount * PAID_PART[execution["contributions"][role]] for role, amount in protocol.contributions.items())


def commissioning(protocol: Protocol, execution: dict[str, Any]) -> tuple[int | None, list[tuple[int, str]]]:
    """Месяц ввода очистных от голосования (None — не введены к году 10) и задержки с причинами."""
    coverage = paid_in(protocol, execution) / protocol.total_cost
    reconstruction = execution["reconstruction"]
    if reconstruction == NONE or coverage < MIN_COVERAGE_TO_BUILD:
        return None, []
    delays = []
    if reconstruction == PARTIAL:
        delays.append((DELAY_PARTIAL_RECONSTRUCTION, "реконструкция выполнена частично"))
    if coverage < 1:
        delays.append((DELAY_UNDERFUNDED, "смета покрыта не полностью"))
    month = protocol.promised_months + sum(months for months, _ in delays)
    return (month, delays) if month <= HORIZON_MONTHS else (None, delays)


def shares(protocol: Protocol) -> dict[str, float]:
    """Доля каждой роли в смете по заявленным взносам."""
    return {role: protocol.contributions.get(role, 0) / protocol.total_cost for role in ROLE_LABELS}


# Исходы проверяются по порядку, срабатывает первый подходящий.
ACCEPTED_RULES = (
    ("nedostroy", lambda f: f["month"] is None),
    ("without_residents", lambda f: not f["support"][RESIDENTS] and not f["support"][NGO]),
    ("dolgostroy", lambda f: f["month"] > f["promised_months"] and WATER not in f["acting"]),
    ("holding_paid", lambda f: f["shares"][HOLDING] >= HOLDING_PAID_ALL_SHARE),
    ("city_paid", lambda f: f["shares"][ADMIN] >= CITY_PAID_ALL_SHARE),
    ("open_river", lambda f: f["month"] == f["promised_months"] and f["monitoring_ok"]
        and all(value == FULL for value in f["levels"]) and f["support"][RESIDENTS] and f["support"][NGO]),
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


def simulate(protocol: Protocol, excluded: str | None = None) -> dict[str, Any]:
    execution = execute(protocol, excluded)
    month, delays = commissioning(protocol, execution)
    # Условие влияет на город, только если выполнено полностью.
    acting = {condition for condition, value in execution["conditions"].items() if value == FULL}
    facts = {
        "month": month,
        "delays": delays,
        "promised_months": protocol.promised_months,
        "paid": paid_in(protocol, execution),
        "contributions": execution["contributions"],
        "conditions": execution["conditions"],
        "levels": [*execution["contributions"].values(), *execution["conditions"].values(),
                   execution["reconstruction"]],
        "acting": acting,
        "included": protocol.included,
        "guilt": protocol.conditions[GUILT] if protocol.accepted else None,
        "support": {role: protocol.supports(role) for role in ROLE_LABELS},
        "shares": shares(protocol),
        "credit": protocol.credit,
        "monitoring_ok": SENSORS in acting,
    }
    return {"execution": execution, "facts": facts, "outcome": choose_outcome(protocol, facts)}


def forecast(game_result: dict[str, Any]) -> dict[str, Any]:
    """Прогноз по итогу игры Влада: Data.data оркестратора."""
    protocol = read_protocol(game_result)
    run = simulate(protocol)
    facts = run["facts"]
    state = city_state(facts)
    without_role = {role: simulate(protocol, excluded=role)["outcome"]["name"] for role in ROLE_LABELS}
    result = {
        "outcome": run["outcome"],
        "protocol": {**asdict(protocol), "credit": protocol.credit, "included": protocol.included},
        "promised_months": protocol.promised_months,
        "paid": facts["paid"],
        "execution": run["execution"],
        "commissioning_month": facts["month"],
        "state": state,
        "fates": fates(facts, state),
        "explanation": {"rules": fired_rules(protocol, facts, state, run["outcome"]), "without_role": without_role},
        "warnings": list(protocol.warnings),
    }
    result["report"] = render_report(result)
    result["herald_prompt"] = render_herald_prompt(result)
    return result
