"""Шаг 2: состояние города — уровни пяти подсистем учебника в момент прогноза."""

from __future__ import annotations

from typing import Any, Callable

from .catalog import (ACHIEVED, ADMIN, CITY_PAID_ALL_SHARE, FAILED, FULL, FUNDING_FROM_OTHERS_SHARE, HOLDING,
                      HOLDING_PAID_ALL_SHARE, HORIZON_MONTHS, LATE_WITHOUT_WATER_MONTHS, MEDIA_HOLDING_MIN_SHARE,
                      MORATORIUM, NGO, NONE, PARTLY, PLANT_MODE, REGION, REGION_LOW_COST_SHARE, RESIDENTS, SENSORS,
                      TARIFFS, TAX_RELIEF, WATER, WATER_MAX_MONTHS, YEAR_1_MONTH)

RANK = {FAILED: 0, PARTLY: 1, ACHIEVED: 2}


def level(failed: bool, achieved: bool) -> str:
    """Сначала «не достигнуто», затем «достигнуто», остальное — «частично»."""
    if failed:
        return FAILED
    return ACHIEVED if achieved else PARTLY


def built_by(facts: dict[str, Any], month: int) -> bool:
    return facts["month"] is not None and facts["month"] <= month


def water_at(facts: dict[str, Any], month: int) -> bool:
    """Гарантия воды: подвоз действует до ввода очистных, но не дольше 4 лет."""
    return WATER in facts["acting"] and month <= WATER_MAX_MONTHS and not built_by(facts, month)


def subsystems(facts: dict[str, Any], month: int, horizon: bool) -> dict[str, str]:
    built = built_by(facts, month)
    residents_have_water = built or water_at(facts, month)
    support, shares, credit = facts["support"], facts["shares"], facts["credit"]
    holding_paid = facts["contributions"].get(HOLDING) == FULL and shares[HOLDING] >= MEDIA_HOLDING_MIN_SHARE
    late_without_water = WATER not in facts["acting"] and (
        facts["month"] is None or facts["month"] > LATE_WITHOUT_WATER_MONTHS)
    return {
        # Учебник, табл. 4, 5: вода в норме и истинное состояние реки известно.
        "ecology": level(not built and not water_at(facts, month), built and facts["monitoring_ok"]),
        # Табл. 5, 7: «Открытая река»; цель холдинга — негатив 90 → 60%.
        "media": level(not holding_paid and not facts["monitoring_ok"], holding_paid and facts["monitoring_ok"]),
        # Табл. 5: занятость, зависимость моногорода от одного работодателя.
        "economy": level(HOLDING in credit,
                         shares[HOLDING] < HOLDING_PAID_ALL_SHARE and shares[ADMIN] < CITY_PAID_ALL_SHARE
                         and not credit),
        # Табл. 5: выявленные и устранённые нарушения, договоры с публичными KPI.
        "law": level(facts["conditions"].get(SENSORS, NONE) == NONE,
                     facts["monitoring_ok"] and all(value == FULL for value in facts["conditions"].values())),
        # Табл. 5; методичка: доверие к власти, протесты.
        "politics": level((not support[RESIDENTS] and not support[NGO]) or late_without_water or (horizon and not built),
                          support[RESIDENTS] and residents_have_water),
    }


def not_below(current: str, floor: str) -> str:
    return current if RANK[current] >= RANK[floor] else floor


def not_above(current: str, cap: str) -> str:
    return current if RANK[current] <= RANK[cap] else cap


def corrections(facts: dict[str, Any], month: int, moment: str) -> list[tuple[str, str, Callable[[str, str], str]]]:
    """Поправки от условий протокола в момент прогноза: (поправка, подсистема, ограничение до «частично»)."""
    found: list[tuple[str, str, Callable[[str, str], str]]] = []
    if PLANT_MODE in facts["acting"] and not built_by(facts, month):
        found += [("plant_mode", "ecology", not_below), ("plant_mode", "economy", not_above)]
    if facts["guilt"] == "holding" and moment == "year_1":
        found.append(("guilt_holding", "media", not_above))
    if facts["guilt"] == "city":
        found.append(("guilt_city", "politics", not_above))
    if TAX_RELIEF in facts["acting"] and HOLDING in facts["credit"]:
        found.append(("tax_relief", "economy", not_below))
    if MORATORIUM in facts["acting"] and moment == "year_1":
        found.append(("protest_moratorium", "politics", not_below))
    return found


def moment_state(facts: dict[str, Any], month: int, moment: str) -> dict[str, Any]:
    levels = subsystems(facts, month, moment == "year_10")
    fired = []
    for correction, subsystem, clamp in corrections(facts, month, moment):
        corrected = clamp(levels[subsystem], PARTLY)
        if corrected != levels[subsystem]:
            levels[subsystem] = corrected
            fired.append(correction)
    return {"month": month, "subsystems": levels, "corrections": list(dict.fromkeys(fired))}


def city_state(facts: dict[str, Any]) -> dict[str, Any]:
    moments = {"year_1": YEAR_1_MONTH, "commissioning": facts["month"], "year_10": HORIZON_MONTHS}
    return {moment: {"month": None, "subsystems": None, "corrections": []} if month is None
            else moment_state(facts, month, moment)
            for moment, month in moments.items()}


def fates(facts: dict[str, Any], state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Судьба участников через 10 лет: проверки по целям ролей из методички."""
    year_1, year_10 = state["year_1"]["subsystems"], state["year_10"]["subsystems"]
    shares, credit = facts["shares"], facts["credit"]

    def check(goal: str, met: bool) -> dict[str, Any]:
        return {"goal": goal, "met": met}

    return {
        ADMIN: [check("Финансирование получено от области и холдинга",
                      shares[REGION] + shares[HOLDING] >= FUNDING_FROM_OTHERS_SHARE),
                check("Массовых протестов нет", year_10["politics"] != FAILED or MORATORIUM in facts["acting"]),
                check("Вину за аварию не возложили на город", facts["guilt"] != "city")],
        HOLDING: [check("Негатив к холдингу снижен до 60% за год", year_1["media"] == ACHIEVED),
                  check("Вложения не чрезмерные", shares[HOLDING] < HOLDING_PAID_ALL_SHARE and HOLDING not in credit)],
        REGION: [check("Затраты области низкие: малая доля и без налоговых льгот холдингу",
                       shares[REGION] <= REGION_LOW_COST_SHARE and TAX_RELIEF not in facts["acting"]),
                 check("Доверие восстановлено: условия для переизбрания губернатора есть",
                       year_10["politics"] == ACHIEVED)],
        RESIDENTS: [check("Чистая вода", year_10["ecology"] != FAILED),
                    check("Тарифы заморожены на время работ", TARIFFS in facts["acting"])],
        NGO: [check("Независимый контроль за очистными", facts["monitoring_ok"]),
              check("Участие в контроле: НКО включила открытые датчики в протокол", SENSORS in facts["included"])],
    }
