"""Шаг 4: какие правила сработали и на что они опираются."""

from __future__ import annotations

from typing import Any

from .catalog import (ADMIN, COMPONENT_LABELS, DELAY_PARTIAL_RECONSTRUCTION, DELAY_UNDERFUNDED, HOLDING,
                      INDEPENDENT_MONITORS, MIN_COVERAGE_TO_BUILD, MONITORING, NGO, PARTIAL, RECONSTRUCTION,
                      RESIDENTS, ROLE_LABELS, SOURCE_FORMALIZATION, SOURCE_METODICHKA,
                      SOURCE_OUTCOMES, SOURCE_TABLE_4, SOURCE_TABLE_4_ANALOGY, SOURCE_TABLE_5_ECONOMY,
                      SOURCE_TABLE_5_LAW, WITHOUT_MEANS)
from .protocol import Protocol

MLN = 1_000_000


def rule(step: str, text: str, source: str) -> dict[str, str]:
    return {"step": step, "text": text, "source": source}


def mln(amount: float) -> str:
    return f"{amount / MLN:g}".replace(".", ",")


def fired_rules(protocol: Protocol, facts: dict[str, Any], outcome: dict[str, str]) -> list[dict[str, str]]:
    rules = []
    owners = set(protocol.contributions) | {owner for owner in protocol.responsible.values() if owner}
    for role in ROLE_LABELS:
        if role not in owners:
            continue
        voted = "«за»" if protocol.votes[role] else "«против»"
        if protocol.accepted:
            result = "выполняются полностью" if protocol.votes[role] else "выполняются частично"
        else:
            result = "заявленное выполняется частично" if protocol.votes[role] else "не выполняются"
            result = "договор не принят: " + result
        rules.append(rule("Исполнение", f"{ROLE_LABELS[role]}: голос {voted} → обязательства {result}.",
                          SOURCE_FORMALIZATION))

    monitor = protocol.responsible[MONITORING]
    if monitor is None:
        rules.append(rule("Исполнение", "За мониторинг никто не отвечает: контроля нет.", SOURCE_FORMALIZATION))
    elif monitor in INDEPENDENT_MONITORS:
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
        elif owner in WITHOUT_MEANS and not (component == MONITORING and owner == NGO):
            who = "жители" if owner == RESIDENTS else "НКО"
            rules.append(rule("Исполнение", f"Компонент «{label}» ведут {who}: нет денег и полномочий → "
                                            "не выше «частично».", SOURCE_METODICHKA))

    paid, total = facts["paid"], protocol.total_cost
    covered = f"Внесено {mln(paid)} из {mln(total)} млн ₽ ({paid / total:.0%})"
    if facts["month"] is not None:
        delays = [f"обещано {facts['promised_months']}"]
        if facts["components"][RECONSTRUCTION] == PARTIAL:
            delays.append(f"+{DELAY_PARTIAL_RECONSTRUCTION} — реконструкция выполнена частично")
        if paid < total:
            delays.append(f"+{DELAY_UNDERFUNDED} — смета покрыта не полностью")
        rules.append(rule("Деньги и сроки", f"{covered} → очистные введены на {facts['month']}-м месяце: "
                                            + ", ".join(delays) + ".", SOURCE_FORMALIZATION))
    elif paid / total < MIN_COVERAGE_TO_BUILD:
        rules.append(rule("Деньги и сроки", f"{covered}, меньше {MIN_COVERAGE_TO_BUILD:.0%} → очистные не введены "
                                            "за 10 лет.", SOURCE_FORMALIZATION))
    else:
        rules.append(rule("Деньги и сроки", f"{covered}, но реконструкция не выполняется или опаздывает дольше "
                                            "10 лет → очистные не введены.", SOURCE_FORMALIZATION))
    for role in facts["credit"]:
        credit = protocol.contributions[role] - protocol.budgets[role]
        rules.append(rule("Деньги и сроки", f"{ROLE_LABELS[role]} вносит сверх бюджета: кредит {mln(credit)} млн ₽ "
                                            "→ долговая нагрузка.", SOURCE_TABLE_5_ECONOMY))

    rules.append(rule("Исход", f"Исход «{outcome['name']}»: {outcome['rule']}.", SOURCE_OUTCOMES))
    return rules
