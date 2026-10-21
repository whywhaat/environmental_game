"""Шаг 2: состояние города — уровни пяти подсистем учебника в момент прогноза."""

from __future__ import annotations

from typing import Any

from .catalog import (ACHIEVED, ADMIN, CITY_PAID_ALL_SHARE, FAILED, FULL, FUNDING_FROM_OTHERS_SHARE, HOLDING,
                      HOLDING_PAID_ALL_SHARE, LATE_WITHOUT_WATER_MONTHS, MEDIA_HOLDING_MIN_SHARE, MONITORING, NGO,
                      HORIZON_MONTHS, NONE, PARTLY, REGION, REGION_LOW_COST_SHARE, RESIDENTS, WATER,
                      WATER_MAX_MONTHS, YEAR_1_MONTH)


def level(failed: bool, achieved: bool) -> str:
    """Сначала «не достигнуто», затем «достигнуто», остальное — «частично»."""
    if failed:
        return FAILED
    return ACHIEVED if achieved else PARTLY


def built_by(facts: dict[str, Any], month: int) -> bool:
    return facts["month"] is not None and facts["month"] <= month


def water_at(facts: dict[str, Any], month: int) -> bool:
    """Подвоз воды и заморозка тарифов действуют до ввода очистных, но не дольше 4 лет."""
    return facts["components"][WATER] == FULL and month <= WATER_MAX_MONTHS and not built_by(facts, month)


def subsystems(facts: dict[str, Any], month: int, horizon: bool) -> dict[str, str]:
    built = built_by(facts, month)
    residents_have_water = built or water_at(facts, month)
    votes, shares, credit = facts["votes"], facts["shares"], facts["credit"]
    holding_paid = facts["contributions"].get(HOLDING) == FULL and shares[HOLDING] >= MEDIA_HOLDING_MIN_SHARE
    late_without_water = facts["components"][WATER] != FULL and (
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
        # Табл. 5: выявленные и устранённые нарушения, прозрачная ответственность.
        "law": level(facts["components"][MONITORING] == NONE,
                     facts["monitoring_ok"] and all(facts["responsible"].values())),
        # Табл. 5; методичка: доверие к власти, протесты.
        "politics": level((not votes[RESIDENTS] and not votes[NGO]) or late_without_water or (horizon and not built),
                          votes[RESIDENTS] and residents_have_water),
    }


def city_state(facts: dict[str, Any]) -> dict[str, Any]:
    moments = {"year_1": YEAR_1_MONTH, "commissioning": facts["month"], "year_10": HORIZON_MONTHS}
    return {moment: {"month": month,
                     "subsystems": None if month is None else subsystems(facts, month, moment == "year_10")}
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
                check("Массовых протестов нет", year_10["politics"] != FAILED)],
        HOLDING: [check("Негатив к холдингу снижен до 60% за год", year_1["media"] == ACHIEVED),
                  check("Вложения не чрезмерные", shares[HOLDING] < HOLDING_PAID_ALL_SHARE and HOLDING not in credit)],
        REGION: [check("Затраты области низкие", shares[REGION] <= REGION_LOW_COST_SHARE),
                 check("Доверие восстановлено: условия для переизбрания губернатора есть",
                       year_10["politics"] == ACHIEVED)],
        RESIDENTS: [check("Чистая вода", year_10["ecology"] != FAILED),
                    check("Тарифы заморожены на время работ", facts["components"][WATER] == FULL)],
        NGO: [check("Независимый контроль за очистными", facts["monitoring_ok"]),
              check("Участие в контроле: мониторинг ведёт НКО", facts["responsible"][MONITORING] == NGO)],
    }
