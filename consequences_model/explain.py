"""Шаг 4: какие правила сработали и на что они опираются."""

from __future__ import annotations

from typing import Any

from .catalog import (ADMIN, COMPONENT_LABELS, HOLDING, MIN_COVERAGE_TO_BUILD, MONITORING, RESIDENTS, ROLE_LABELS,
                      SOURCE_FORMALIZATION, SOURCE_METODICHKA, SOURCE_OUTCOMES, SOURCE_TABLE_4,
                      SOURCE_TABLE_4_ANALOGY, SOURCE_TABLE_5_ECONOMY, SOURCE_TABLE_5_LAW, SUBSYSTEM_CRITERIA,
                      SUBSYSTEM_LABELS, VOTE_LABELS)
from .protocol import Protocol
from .report import millions


def rule(step: str, text: str, source: str) -> dict[str, str]:
    return {"step": step, "text": text, "source": source}


def execution_rules(protocol: Protocol, facts: dict[str, Any]) -> list[dict[str, str]]:
    rules = []
    owners = set(protocol.contributions) | {owner for owner in protocol.responsible.values() if owner}
    for role in ROLE_LABELS:
        if role not in owners:
            continue
        voted = f"«{VOTE_LABELS[protocol.votes[role]]}»"
        if protocol.accepted:
            text = f"голос {voted} → обязательства " + ("выполняются полностью" if protocol.supports(role)
                                                       else "выполняются частично")
        else:
            text = f"голос {voted}, протокол не принят → " + ("заявленное выполняется частично" if protocol.supports(role)
                                                             else "обязательства не выполняются")
        rules.append(rule("Исполнение", f"{ROLE_LABELS[role]}: {text}.", SOURCE_FORMALIZATION))

    monitor = protocol.responsible[MONITORING]
    if monitor is None:
        rules.append(rule("Исполнение", "За мониторинг никто не отвечает: контроля нет.", SOURCE_FORMALIZATION))
    elif protocol.monitoring_is_independent:
        rules.append(rule("Исполнение", f"Мониторинг ведёт {ROLE_LABELS[monitor]} — независимая лаборатория.",
                          SOURCE_TABLE_4))
    elif monitor == HOLDING:
        rules.append(rule("Исполнение", f"Мониторинг ведёт {ROLE_LABELS[HOLDING]} — «своя лаборатория»: "
                                        "истинное состояние реки неизвестно.", SOURCE_TABLE_4))
    elif monitor == ADMIN:
        rules.append(rule("Исполнение", f"Мониторинг ведёт {ROLE_LABELS[ADMIN]} — владелец очистных проверяет "
                                        "сам себя.", SOURCE_TABLE_4_ANALOGY))
    if protocol.accepted and facts["monitoring_ok"]:
        rules.append(rule("Исполнение", "Независимый мониторинг выполнен полностью → срывы видны и исправляются: "
                                        "остальные обязательства поднимаются на ступень.", SOURCE_TABLE_5_LAW))
    for component, owner in protocol.responsible.items():
        label = COMPONENT_LABELS[component]
        if owner is None:
            rules.append(rule("Исполнение", f"За компонент «{label}» никто не отвечает → он не выполняется.",
                              SOURCE_FORMALIZATION))
        elif component in facts["capped"]:
            who = "жители" if owner == RESIDENTS else "НКО"
            rules.append(rule("Исполнение", f"Компонент «{label}» ведут {who}: нет денег и полномочий → "
                                            "не выше «частично».", SOURCE_METODICHKA))
    return rules


def money_rules(protocol: Protocol, facts: dict[str, Any]) -> list[dict[str, str]]:
    paid, total = facts["paid"], protocol.total_cost
    covered = f"Внесено {millions(paid)} из {millions(total)} млн ₽ ({paid / total:.0%})"
    if facts["month"] is not None:
        delays = [f"обещано {facts['promised_months']}"] + [f"+{months} — {reason}" for months, reason in facts["delays"]]
        text = f"{covered} → очистные введены на {facts['month']}-м месяце: " + ", ".join(delays) + "."
    elif paid / total < MIN_COVERAGE_TO_BUILD:
        text = f"{covered}, меньше {MIN_COVERAGE_TO_BUILD:.0%} → очистные не введены за 10 лет."
    else:
        text = f"{covered}, но реконструкция не выполняется или опаздывает дольше 10 лет → очистные не введены."
    rules = [rule("Деньги и сроки", text, SOURCE_FORMALIZATION)]
    for role in facts["credit"]:
        credit = protocol.contributions[role] - protocol.budgets[role]
        rules.append(rule("Деньги и сроки", f"{ROLE_LABELS[role]} вносит сверх бюджета: кредит {millions(credit)} "
                                            "млн ₽ → долговая нагрузка.", SOURCE_TABLE_5_ECONOMY))
    return rules


def state_rules(state: dict[str, Any]) -> list[dict[str, str]]:
    year_10 = state["year_10"]["subsystems"]
    rules = []
    for subsystem, label in SUBSYSTEM_LABELS.items():
        source, criteria = SUBSYSTEM_CRITERIA[subsystem]
        level = year_10[subsystem]
        rules.append(rule("Состояние города", f"{label} через 10 лет — {level}: {criteria[level]}.", source))
    return rules


def fired_rules(protocol: Protocol, facts: dict[str, Any], state: dict[str, Any],
                outcome: dict[str, str]) -> list[dict[str, str]]:
    return (execution_rules(protocol, facts) + money_rules(protocol, facts) + state_rules(state)
            + [rule("Исход", f"Исход «{outcome['name']}»: {outcome['rule']}.", SOURCE_OUTCOMES)])
