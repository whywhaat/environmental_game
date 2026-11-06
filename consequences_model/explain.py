"""Шаг 4: какие правила сработали и на что они опираются."""

from __future__ import annotations

from typing import Any

from .catalog import (CONDITIONS, CORRECTIONS, FULL, MIN_COVERAGE_TO_BUILD, PARTIAL, RECONSTRUCTION_OWNER,
                      ROLE_LABELS, SENSORS, SOURCE_FORMALIZATION, SOURCE_OUTCOMES, SOURCE_TABLE_4,
                      SOURCE_TABLE_5_ECONOMY, SOURCE_TABLE_5_LAW, SUBSYSTEM_CRITERIA, SUBSYSTEM_LABELS, VOTE_LABELS)
from .protocol import Protocol
from .report import millions


def rule(step: str, text: str, source: str) -> dict[str, str]:
    return {"step": step, "text": text, "source": source}


def execution_rules(protocol: Protocol, facts: dict[str, Any]) -> list[dict[str, str]]:
    rules = []
    owners = {RECONSTRUCTION_OWNER, *protocol.contributions,
              *(CONDITIONS[condition]["owner"] for condition in protocol.included)}
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

    for condition, value in facts["conditions"].items():
        if value != FULL:
            done = "выполнено частично" if value == PARTIAL else "не выполнено"
            rules.append(rule("Исполнение", f"Условие «{CONDITIONS[condition]['label']}» {done} → на город не влияет.",
                              SOURCE_FORMALIZATION))

    if SENSORS not in protocol.included:
        rules.append(rule("Исполнение", "Открытые датчики не включены в протокол: независимого контроля нет, "
                                        "истинное состояние реки неизвестно.", SOURCE_TABLE_4))
    elif protocol.accepted and facts["monitoring_ok"]:
        rules.append(rule("Исполнение", "Открытые датчики НКО выполнены полностью → срывы видны и исправляются: "
                                        "остальные обязательства поднимаются на ступень.", SOURCE_TABLE_5_LAW))
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
        rules.append(rule("Деньги и сроки", f"{ROLE_LABELS[role]} вносит сверх бюджета: кредит "
                                            f"{millions(protocol.overspend(role))} "
                                            "млн ₽ → долговая нагрузка.", SOURCE_TABLE_5_ECONOMY))
    return rules


def state_rules(state: dict[str, Any]) -> list[dict[str, str]]:
    year_10 = state["year_10"]["subsystems"]
    rules = []
    for subsystem, label in SUBSYSTEM_LABELS.items():
        source, criteria = SUBSYSTEM_CRITERIA[subsystem]
        level = year_10[subsystem]
        correction = state["year_10"]["corrected"].get(subsystem)
        if correction:
            rules.append(rule("Состояние города", f"{label} через 10 лет — {level}: уровень задан условием "
                                                  "протокола, см. поправку ниже.", CORRECTIONS[correction][1]))
        else:
            rules.append(rule("Состояние города", f"{label} через 10 лет — {level}: {criteria[level]}.", source))
    return rules


def correction_rules(state: dict[str, Any]) -> list[dict[str, str]]:
    fired = dict.fromkeys(correction for moment in state.values() for correction in moment["corrected"].values())
    return [rule("Поправки условий", f"{CORRECTIONS[correction][0]}.", CORRECTIONS[correction][1])
            for correction in fired]


def fired_rules(protocol: Protocol, facts: dict[str, Any], state: dict[str, Any],
                outcome: dict[str, str]) -> list[dict[str, str]]:
    return (execution_rules(protocol, facts) + money_rules(protocol, facts) + state_rules(state)
            + correction_rules(state)
            + [rule("Исход", f"Исход «{outcome['name']}»: {outcome['rule']}.", SOURCE_OUTCOMES)])
