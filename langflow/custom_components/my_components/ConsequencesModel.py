"""Модель последствий для Langflow. СОБРАНО из пакета consequences_model — руками не править.

Пересобрать: python -m consequences_model.build_component
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Any, Callable
from dataclasses import asdict
from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


# --- consequences_model/catalog.py: Данные модели: роли, условия протокола, уровни, пороги и каталог исходов.

ADMIN = "city_administration"
HOLDING = "industrial_holding"
REGION = "regional_government"
RESIDENTS = "city_residents"
NGO = "environmental_ngo"

ROLE_LABELS = {
    ADMIN: "Администрация города N",
    HOLDING: "Промышленный холдинг",
    REGION: "Региональная власть",
    RESIDENTS: "Жители города N",
    NGO: "Экологическая организация (НКО)",
}

# Условия протокола игры Влада: у каждого роль-владелец, только она включает условие или выбирает значение.
TIMELINE = "timeline"
PLANT_MODE = "plant_mode"
GUILT = "guilt"
TARIFFS = "tariffs_freeze"
TAX_RELIEF = "tax_relief"
SENSORS = "open_sensors"
MORATORIUM = "protest_moratorium"
WATER = "water_supply"

CONDITIONS = {
    TIMELINE: {"label": "Срок реконструкции", "owner": HOLDING},
    PLANT_MODE: {"label": "Режим работы комбината", "owner": HOLDING},
    GUILT: {"label": "Признание вины за аварию", "owner": REGION},
    TARIFFS: {"label": "Заморозка тарифов ЖКХ", "owner": ADMIN},
    TAX_RELIEF: {"label": "Налоговые льготы холдингу", "owner": REGION},
    SENSORS: {"label": "Открытые эко-датчики 24/7", "owner": NGO},
    MORATORIUM: {"label": "Мораторий на протесты и иски", "owner": RESIDENTS},
    WATER: {"label": "Гарантия питьевой воды", "owner": ADMIN},
}
# Значения по умолчанию — как в игре: они стоят в проекте протокола, пока владелец их не поменял.
CONDITION_DEFAULTS = {TIMELINE: 36, PLANT_MODE: "100%", GUILT: "shared", TARIFFS: False, TAX_RELIEF: False,
                      SENSORS: False, MORATORIUM: False, WATER: False}
TIMELINE_MONTHS = (24, 48)
PLANT_MODES = ("100%", "70%")
PLANT_REDUCED = "70%"
GUILT_ON_HOLDING = "holding"
GUILT_ON_CITY = "city"
GUILT_LABELS = {"shared": "обоюдная", GUILT_ON_HOLDING: "на холдинге", GUILT_ON_CITY: "на городе"}
# Включённые «да/нет»-условия — обязательства владельца; режим комбината — только при 70%.
YES_NO_CONDITIONS = (WATER, TARIFFS, TAX_RELIEF, SENSORS, MORATORIUM)
RECONSTRUCTION_LABEL = "Реконструкция очистных"
RECONSTRUCTION_OWNER = HOLDING  # владелец срока реконструкции

# Голос по протоколу: None — воздержался (или не голосовал).
VOTE_LABELS = {True: "за", False: "против", None: "воздержался"}
GAME_VOTES = {"ЗА": True, "ПРОТИВ": False, "ВОЗДЕРЖАЛСЯ": None}

# Исполнение обязательства.
FULL = "полностью"
PARTIAL = "частично"
NONE = "нет"

# Пороги — наша формализация; калибруются после тестового прогона.
PAID_PART = {FULL: 1.0, PARTIAL: 0.5, NONE: 0.0}  # какая часть взноса реально внесена
MIN_COVERAGE_TO_BUILD = 0.75
DELAY_PARTIAL_RECONSTRUCTION = 12  # мес.
DELAY_UNDERFUNDED = 24  # мес.
HORIZON_MONTHS = 120

RAISE = {NONE: PARTIAL, PARTIAL: FULL, FULL: FULL}

# Каталог исходов: название, условие словами (для объяснения) и описание для отчёта и «Вестника».
OUTCOMES = {
    "nedostroy": {
        "name": "Недострой",
        "rule": "очистные не введены к году 10",
        "description": "Через 10 лет очистные так и не введены: реконструкцию недофинансировали или выполнили не "
                       "полностью. Река остаётся загрязнённой, временные меры к этому "
                       "времени свёрнуты, конфликт вокруг очистных продолжается.",
    },
    "without_residents": {
        "name": "Решили без жителей",
        "rule": "жители и НКО не поддержали протокол: голосовали против или воздержались",
        "description": "Протокол приняли без жителей и экологов: они его не поддержали. Работы идут, но решение "
                       "воспринимается как навязанное: недовольство и протестная активность сохраняются, "
                       "экологическая тема политизирована (учебник, табл. 5, «Политика»).",
    },
    "dolgostroy": {
        "name": "Долгострой",
        "rule": "очистные введены позже срока из протокола, а гарантия питьевой воды не действует",
        "description": "Очистные введены позже срока из протокола, а подвоз воды на время работ не был "
                       "обеспечен полностью. Годы ожидания с перебоями в воде подорвали доверие к "
                       "исполнителям, недовольство жителей выросло.",
    },
    "holding_paid": {
        "name": "Холдинг заплатил за всех",
        "rule": "доля холдинга в смете не меньше 70%",
        "description": "Почти все расходы легли на холдинг: цель области переложить затраты на холдинг "
                       "достигнута. Градообразующее предприятие перегружено, растёт давление на рабочие места, "
                       "зависимость моногорода от одного работодателя усиливается (учебник, табл. 5, «Экономика»).",
    },
    "city_paid": {
        "name": "Город заплатил за всех",
        "rule": "доля администрации в смете не меньше 50%",
        "description": "Основную долю сметы взяла на себя администрация города. Городской бюджет истощён, "
                       "на другие задачи средств не остаётся, конфликт администрации с холдингом продолжается.",
    },
    "open_river": {
        "name": "Открытая река",
        "rule": "очистные введены в срок из протокола, все взносы и включённые условия выполнены полностью, "
                "открытые датчики работают, жители и НКО голосовали за",
        "description": "Очистные введены в срок из протокола, а открытые эко-датчики НКО показывают всем "
                       "состояние реки. Затраты разделены, жители и экологи участвовали в "
                       "решении, негатив к холдингу снижается. Это решение близко к эталонному "
                       "(учебник, табл. 7).",
    },
    "closed_data": {
        "name": "Чистая вода, закрытые данные",
        "rule": "открытые датчики не включены в протокол или выполнены не полностью",
        "description": "Очистные работают, но открытых данных о реке нет: эко-датчики не поставлены или "
                       "работают не полностью, и истинное состояние реки неизвестно (учебник, табл. 4). Жители не "
                       "верят отчётам, недоверие держится.",
    },
    "compromise": {
        "name": "Компромисс",
        "rule": "запасной исход для принятого протокола",
        "description": "Очистные введены, но протокол далёк от эталонного решения: часть условий выполнена не "
                       "полностью или без поддержки части участников. Ситуация лучше, чем после аварии, но "
                       "доверие и экология восстановлены лишь отчасти.",
    },
    "each_for_self": {
        "name": "Каждый сам за себя",
        "rule": "протокол не принят, но хотя бы одно обязательство выполнено частично",
        "description": "Общий протокол не принят. Отдельные стороны частично выполняют то, что заявляли, но без "
                       "общего плана и контроля. Улучшения медленные и частичные, ответственность размыта.",
    },
    "stagnation": {
        "name": "Застой",
        "rule": "протокол не принят, и никто ничего не выполняет",
        "description": "Протокол не принят, и никто не выполняет своих обязательств. Очистные не реконструированы, "
                       "река остаётся загрязнённой, недовольство жителей растёт, конфликт администрации и "
                       "холдинга продолжается.",
    },
}

HOLDING_PAID_ALL_SHARE = 0.70
CITY_PAID_ALL_SHARE = 0.50

# Уровни подсистем (учебник, табл. 5).
ACHIEVED = "достигнуто"
PARTLY = "частично"
FAILED = "не достигнуто"

SUBSYSTEM_LABELS = {
    "ecology": "Экология и наука",
    "media": "Массмедиа",
    "economy": "Экономика",
    "law": "Право",
    "politics": "Политика",
}

MOMENT_LABELS = {"year_1": "Через год", "commissioning": "При вводе очистных", "year_10": "Через 10 лет"}
YEAR_1_MONTH = 12
WATER_MAX_MONTHS = 48  # подвоз воды — до ввода очистных, но не дольше 4 лет
LATE_WITHOUT_WATER_MONTHS = 36
MEDIA_HOLDING_MIN_SHARE = 0.30

FUNDING_FROM_OTHERS_SHARE = 0.70  # администрация получила деньги от области и холдинга
REGION_LOW_COST_SHARE = 0.20

# Источники правил для объяснения.
SOURCE_FORMALIZATION = "наша формализация"
SOURCE_TABLE_4 = "Учебник, 3.1, табл. 4"
SOURCE_TABLE_5_LAW = "Учебник, 3.1, табл. 5 («Право»)"
SOURCE_TABLE_5_ECONOMY = "Учебник, 3.1, табл. 5 («Экономика»)"
SOURCE_OUTCOMES = "каталог исходов (наша формализация)"

SOURCE_TABLES_4_5 = "Учебник, 3.1, табл. 4, 5"
SOURCE_TABLES_5_7 = "Учебник, 3.1, табл. 5, 7"
SOURCE_TABLE_5_POLITICS = "Учебник, 3.1, табл. 5 («Политика»); методичка"

# Поправки от условий протокола к уровням подсистем: текст для объяснения и опора.
GUILT_HOLDING_CORRECTION = "guilt_holding"
GUILT_CITY_CORRECTION = "guilt_city"
CORRECTIONS = {
    PLANT_MODE: (
        "Режим комбината 70% до ввода очистных: «Экология и наука» не ниже «частично» — сбросов меньше; "
        "«Экономика» не выше «частично» — падают занятость и выручка",
        "наша формализация по учебнику, 3.1, табл. 5 («Экономика»: уровень занятости)"),
    GUILT_HOLDING_CORRECTION: (
        "Вина за аварию признана за холдингом: «Массмедиа» через год не выше «частично» — негатив к холдингу не "
        "упадёт до 60%, если вина официально на нём",
        "наша формализация по учебнику, 3.1, табл. 5 («Массмедиа»: негативный имидж компании)"),
    GUILT_CITY_CORRECTION: (
        "Вина за аварию признана за городом: «Политика» не выше «частично» — подорвано доверие к местной власти",
        "наша формализация по учебнику, 3.1, табл. 5 («Политика»: индекс доверия к местной власти)"),
    TAX_RELIEF: (
        "Налоговые льготы холдингу: его кредит не опускает «Экономику» ниже «частично»",
        "наша формализация по учебнику, 3.1, табл. 5 («Экономика»: налоговые поступления)"),
    MORATORIUM: (
        "Мораторий на протесты — мера на время работ: до ввода очистных «Политика» не ниже «частично»; через 10 лет "
        "он не учитывается",
        "наша формализация по учебнику, 3.1, табл. 5 («Политика»: число публичных конфликтов)"),
}

# Почему подсистема получила уровень: критерий словами и опора.
SUBSYSTEM_CRITERIA = {
    "ecology": (SOURCE_TABLES_4_5, {
        ACHIEVED: "очистные введены, открытые датчики выполнены полностью",
        PARTLY: "очистные введены без открытых датчиков или жителям пока возят воду",
        FAILED: "нет ни очистных, ни подвоза воды"}),
    "media": (SOURCE_TABLES_5_7, {
        ACHIEVED: "холдинг полностью внёс заметную долю (от 30%), а открытые датчики публикуют данные",
        PARTLY: "есть только одно из двух: заметный вклад холдинга или открытые данные",
        FAILED: "нет ни заметного вклада холдинга, ни открытых данных"}),
    "economy": (SOURCE_TABLE_5_ECONOMY, {
        ACHIEVED: "нагрузка не перекошена: холдинг меньше 70%, администрация меньше 50%, кредитов нет",
        PARTLY: "нагрузка смещена на холдинг или город либо кто-то платит в кредит",
        FAILED: "холдинг платит в кредит"}),
    "law": (SOURCE_TABLE_5_LAW, {
        ACHIEVED: "открытые датчики и все включённые условия выполнены полностью",
        PARTLY: "датчики работают не полностью или не все включённые условия выполнены",
        FAILED: "открытые датчики не включены в протокол или не выполняются"}),
    "politics": (SOURCE_TABLE_5_POLITICS, {
        ACHIEVED: "жители голосовали «за», и вода у них есть",
        PARTLY: "жители не поддержали протокол или остались без надёжной воды, но до массовых протестов не дошло",
        FAILED: "жители и НКО не поддержали протокол, или очистных нет, или жители долго без воды"}),
}


# --- consequences_model/protocol.py: Адаптер: итог игры Влада → Протокол.

@dataclass(frozen=True)
class Protocol:
    total_cost: float
    contributions: dict[str, float]
    conditions: dict[str, Any]
    votes: dict[str, bool | None]
    accepted: bool
    budgets: dict[str, float] = field(default_factory=dict)
    spent_on_conditions: dict[str, float] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default=())

    def supports(self, role: str) -> bool:
        """Роль поддержала протокол: голос «за». «Против» и «воздержался» — не поддержка."""
        return self.votes[role] is True

    @property
    def included(self) -> list[str]:
        """Условия, которые стали обязательствами владельцев: включённые «да/нет» и режим комбината 70%."""
        return [condition for condition in CONDITIONS
                if (condition in YES_NO_CONDITIONS and self.conditions[condition] is True)
                or (condition == PLANT_MODE and self.conditions[condition] == PLANT_REDUCED)]

    @property
    def promised_months(self) -> int:
        return int(self.conditions[TIMELINE])

    def overspend(self, role: str) -> float:
        """Насколько взнос роли вместе с расходами на условия больше её бюджета."""
        return self.contributions.get(role, 0) + self.spent_on_conditions.get(role, 0) - self.budgets[role]

    @property
    def credit(self) -> list[str]:
        """Роли, чей взнос вместе с расходами на условия больше бюджета: превышение покрыто кредитом."""
        return [role for role in ROLE_LABELS if role in self.budgets and self.overspend(role) > 0]


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
    contributions, spent_on_conditions = {}, {}
    for name, record in (state.get("players") or {}).items():
        role = role_of(name)
        if role and record.get("contribution"):
            contributions[role] = record["contribution"]
        if role and record.get("spent_on_levers"):
            spent_on_conditions[role] = record["spent_on_levers"]

    game_conditions = state.get("levers") or {}
    conditions = {}
    for condition, default in CONDITION_DEFAULTS.items():
        value = game_conditions.get(condition, default)
        if not valid(condition, value):
            warnings.append(f"Недопустимое значение {value!r} условия «{CONDITIONS[condition]['label']}»: "
                            "взято значение по умолчанию.")
            value = default
        conditions[condition] = value
    for condition in game_conditions:
        if condition not in CONDITIONS:
            warnings.append(f"Неизвестное условие {condition}: в прогнозе не учитывается.")

    # Игрок без записи о голосе считается воздержавшимся.
    votes: dict[str, bool | None] = {role: None for role in ROLE_LABELS}
    for record in (game_result.get("voting_summary") or {}).get("records") or []:
        role = role_of(record.get("player"))
        if role and record.get("vote") not in GAME_VOTES:
            warnings.append(f"Неизвестный голос «{record.get('vote')}» у игрока «{record.get('player')}»: считается "
                            "«воздержался».")
        if role:
            votes[role] = GAME_VOTES.get(record.get("vote"))

    return Protocol(game_result["total_cost"], contributions, conditions, votes,
                    bool(game_result.get("consensus_reached")), budgets, spent_on_conditions,
                    tuple(dict.fromkeys(warnings)))


# --- consequences_model/state.py: Шаг 2: состояние города — уровни пяти подсистем учебника в момент прогноза.

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
        # Табл. 5: выявленные и устранённые нарушения, «договоры с публичными KPI».
        "law": level(facts["conditions"].get(SENSORS, NONE) == NONE,
                     facts["monitoring_ok"] and all(value == FULL for value in facts["conditions"].values())),
        # Табл. 5; методичка: доверие к власти, протесты.
        "politics": level((not support[RESIDENTS] and not support[NGO]) or late_without_water or (horizon and not built),
                          support[RESIDENTS] and residents_have_water),
    }


def at_least_partly(level: str) -> str:
    return level if RANK[level] >= RANK[PARTLY] else PARTLY


def at_most_partly(level: str) -> str:
    return level if RANK[level] <= RANK[PARTLY] else PARTLY


def corrections(facts: dict[str, Any], month: int, moment: str) -> list[tuple[str, str, Callable[[str], str]]]:
    """Поправки от условий протокола в момент прогноза: (поправка, подсистема, ограничение до «частично»)."""
    found: list[tuple[str, str, Callable[[str], str]]] = []
    if PLANT_MODE in facts["acting"] and not built_by(facts, month):
        found += [(PLANT_MODE, "ecology", at_least_partly), (PLANT_MODE, "economy", at_most_partly)]
    if facts["guilt"] == GUILT_ON_HOLDING and moment == "year_1":
        found.append((GUILT_HOLDING_CORRECTION, "media", at_most_partly))
    if facts["guilt"] == GUILT_ON_CITY:
        found.append((GUILT_CITY_CORRECTION, "politics", at_most_partly))
    if TAX_RELIEF in facts["acting"] and HOLDING in facts["credit"]:
        found.append((TAX_RELIEF, "economy", at_least_partly))
    # Мораторий — мера на время работ: до ввода очистных, но не в горизонте прогноза.
    if MORATORIUM in facts["acting"] and moment != "year_10" and not built_by(facts, month):
        found.append((MORATORIUM, "politics", at_least_partly))
    return found


def moment_state(facts: dict[str, Any], month: int, moment: str) -> dict[str, Any]:
    """Уровни подсистем в момент прогноза и какие из них задала поправка условия: {подсистема: поправка}."""
    levels = subsystems(facts, month, moment == "year_10")
    corrected = {}
    for correction, subsystem, clamp in corrections(facts, month, moment):
        if clamp(levels[subsystem]) != levels[subsystem]:
            levels[subsystem] = clamp(levels[subsystem])
            corrected[subsystem] = correction
    return {"month": month, "subsystems": levels, "corrected": corrected}


def city_state(facts: dict[str, Any]) -> dict[str, Any]:
    moments = {"year_1": YEAR_1_MONTH, "commissioning": facts["month"], "year_10": HORIZON_MONTHS}
    return {moment: {"month": None, "subsystems": None, "corrected": {}} if month is None
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
                check("Массовых протестов нет", year_10["politics"] != FAILED),
                check("Вину за аварию не возложили на город", facts["guilt"] != GUILT_ON_CITY)],
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


# --- consequences_model/report.py: Шаг 5: отчёт по шаблону (без LLM) и промпт для «Вестника города N, 2036».

MLN = 1_000_000

# Показатели критерия 6: из какой подсистемы через 10 лет и как называть её уровень.
INDICATORS = (
    ("Экологический риск", "ecology", {ACHIEVED: "низкий", PARTLY: "средний", FAILED: "высокий"}),
    ("Недовольство и риск протестов", "politics", {ACHIEVED: "низкие", PARTLY: "средние", FAILED: "высокие"}),
    ("Доверие к администрации", "politics", {ACHIEVED: "восстановлено", PARTLY: "восстановлено частично",
                                             FAILED: "подорвано"}),
    ("Негатив к холдингу", "media", {ACHIEVED: "снижен до 60% и ниже", PARTLY: "снижается, но выше 60%",
                                     FAILED: "остаётся около 90%"}),
    ("Рабочие места", "economy", {ACHIEVED: "сохранены", PARTLY: "под давлением", FAILED: "под угрозой"}),
)


def millions(amount: float) -> str:
    """Сумма в миллионах без единиц: 225 000 000 → «225»."""
    return f"{amount / MLN:g}".replace(".", ",")


def mln(amount: float) -> str:
    return millions(amount) + " млн ₽"


def terms(result: dict[str, Any]) -> str:
    month, promised = result["commissioning_month"], result["promised_months"]
    if month is None:
        return f"обещано {promised} мес., очистные не введены за 10 лет"
    return f"обещано {promised} мес., очистные введены на {month}-м месяце"


def events(result: dict[str, Any]) -> str:
    month, promised = result["commissioning_month"], result["promised_months"]
    if month is None:
        happened = ["очистные не введены за 10 лет"]
    elif month > promised:
        happened = [f"очистные введены на {month}-м месяце, на {month - promised} мес. позже обещанного"]
    else:
        happened = [f"очистные введены в обещанный срок, на {month}-м месяце"]
    happened += [f"{ROLE_LABELS[role]} платит в кредит" for role in result["protocol"]["credit"]]
    return "; ".join(happened)


def condition_value(condition: str, value: Any) -> str:
    if condition == GUILT:
        return GUILT_LABELS[value]
    if isinstance(value, bool):
        return "включено" if value else "не включено"
    return str(value)


def condition_execution(result: dict[str, Any], condition: str) -> str:
    if condition == GUILT:
        return "решение: действует, если протокол принят"
    return result["execution"]["conditions"].get(condition, "—")


def temporary_measures(result: dict[str, Any]) -> str:
    conditions = result["execution"]["conditions"]
    water = conditions.get(WATER, "не включена")
    tariffs = conditions.get(TARIFFS, "не включена")
    return f"гарантия питьевой воды — {water}, подвоз до ввода очистных, не дольше {WATER_MAX_MONTHS // 12} лет; " \
           f"заморозка тарифов на время работ — {tariffs}"


def protocol_conditions(result: dict[str, Any]) -> str:
    if not result["protocol"]["accepted"]:
        return "протокол не принят, его условия не вступили в силу"
    conditions = result["protocol"]["conditions"]
    included = [f"{CONDITIONS[condition]['label'][0].lower()}{CONDITIONS[condition]['label'][1:]} — {level}"
                for condition, level in result["execution"]["conditions"].items()]
    return (f"срок реконструкции {conditions[TIMELINE]} мес.; режим работы комбината {conditions[PLANT_MODE]}; "
            f"вина за аварию {GUILT_LABELS[conditions[GUILT]]}; "
            + ("включены: " + ", ".join(included) if included else "других условий не включено"))


def cost_split(result: dict[str, Any]) -> str:
    protocol = result["protocol"]
    total = protocol["total_cost"]
    parts = [f"{ROLE_LABELS[role]} — {mln(amount)} ({amount / total:.0%})"
             for role, amount in protocol["contributions"].items()]
    text = "; ".join(parts) if parts else "взносов нет"
    text += f". Фактически внесено {mln(result['paid'])} из {mln(total)}"
    if protocol["credit"]:
        text += ". В кредите: " + ", ".join(ROLE_LABELS[role] for role in protocol["credit"])
    return text


def render_report(result: dict[str, Any]) -> str:
    outcome, protocol, execution = result["outcome"], result["protocol"], result["execution"]
    lines = ["# Модель последствий: город N через 10 лет", "",
             f"**Исход: «{outcome['name']}».** {outcome['description']}", "",
             f"Почему: {outcome['rule']}.", "",
             "## Кто что сделал", "",
             "| Роль | Голос | Взнос | Исполнение взноса |", "| --- | --- | --- | --- |"]
    for role, label in ROLE_LABELS.items():
        amount = protocol["contributions"].get(role)
        lines.append(f"| {label} | {VOTE_LABELS[protocol['votes'][role]]} | "
                     f"{mln(amount) if amount else '—'} | {execution['contributions'].get(role, '—')} |")
    lines += ["", "| Условие протокола | Значение | Владелец | Исполнение |", "| --- | --- | --- | --- |",
              f"| {RECONSTRUCTION_LABEL} | срок {protocol['conditions'][TIMELINE]} мес. | "
              f"{ROLE_LABELS[RECONSTRUCTION_OWNER]} | {execution['reconstruction']} |"]
    for condition, info in CONDITIONS.items():
        if condition == TIMELINE:
            continue
        lines.append(f"| {info['label']} | {condition_value(condition, protocol['conditions'][condition])} | "
                     f"{ROLE_LABELS[info['owner']]} | {condition_execution(result, condition)} |")

    state = result["state"]
    headers = [MOMENT_LABELS[moment] + (f" ({state[moment]['month']}-й мес.)" if moment == "commissioning"
                                        and state[moment]["month"] else "") for moment in state]
    lines += ["", "## Состояние города", "", "| Подсистема | " + " | ".join(headers) + " |",
              "| --- |" + " --- |" * len(state)]
    for subsystem, label in SUBSYSTEM_LABELS.items():
        cells = [moment["subsystems"][subsystem] if moment["subsystems"] else "очистные не введены"
                 for moment in state.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    year_10 = state["year_10"]["subsystems"]
    lines += ["", "## Показатели критерия 6", "",
              f"- **Распределение затрат:** {cost_split(result)}.",
              f"- **Сроки:** {terms(result)}."]
    lines += [f"- **{name}:** {words[year_10[subsystem]]}." for name, subsystem, words in INDICATORS]
    lines.append(f"- **Доступность временных мер:** {temporary_measures(result)}.")

    lines += ["", "## Судьбы участников", ""]
    for role, checks in result["fates"].items():
        marks = "; ".join(("✓ " if check["met"] else "✗ ") + check["goal"] for check in checks)
        lines.append(f"- **{ROLE_LABELS[role]}:** {marks}.")

    lines += ["", "## Почему так", ""]
    lines += [f"{number}. {rule['text']} _({rule['source']})_"
              for number, rule in enumerate(result["explanation"]["rules"], 1)]
    lines += ["", "## Без вклада роли", "", "| Если бы не выполнила свои обязательства | Исход |", "| --- | --- |"]
    lines += [f"| {ROLE_LABELS[role]} | {name} |" for role, name in result["explanation"]["without_role"].items()]
    lines += ["", "## Допущения", "",
              "Пороги правил, коэффициенты исполнения и влияние условий протокола — наша формализация, требуют "
              "согласования с заказчицей. "
              "Прогноз — учебная гипотеза, а не прогноз по реальной реке."]
    if result["warnings"]:
        lines += ["", "## Предупреждения", ""] + [f"- {warning}" for warning in result["warnings"]]
    return "\n".join(lines)


def render_herald_prompt(result: dict[str, Any]) -> str:
    outcome, state = result["outcome"], result["state"]
    year_10 = state["year_10"]["subsystems"]
    facts = [f"Исход: «{outcome['name']}». {outcome['description']}",
             f"Условия протокола: {protocol_conditions(result)}.",
             f"Сроки: {terms(result)}.",
             f"Распределение затрат: {cost_split(result)}."]
    for moment, label in MOMENT_LABELS.items():
        month, levels = state[moment]["month"], state[moment]["subsystems"]
        if levels is None:
            facts.append(f"{label}: очистные не введены за 10 лет.")
            continue
        heading = f"{label} ({month}-й мес.)" if moment == "commissioning" else label
        facts.append(f"{heading}: " + "; ".join(f"{name} — {levels[subsystem]}"
                                               for subsystem, name in SUBSYSTEM_LABELS.items()) + ".")
    facts += [f"События: {events(result)}.", f"Временные меры: {temporary_measures(result)}."]
    facts += [f"{name}: {words[year_10[subsystem]]}." for name, subsystem, words in INDICATORS]
    for role, checks in result["fates"].items():
        facts.append(f"{ROLE_LABELS[role]}: " + "; ".join(
            ("достигнуто — " if check["met"] else "не достигнуто — ") + check["goal"] for check in checks) + ".")
    return "\n".join([
        "Напиши короткую заметку для газеты «Вестник города N, 2036» о том, каким стал город N через 10 лет "
        "после общественных слушаний по реконструкции очистных сооружений.",
        "",
        "Правила:",
        "- Опирайся только на факты ниже. Не добавляй фактов, событий и чисел, которых нет в списке.",
        "- Не утверждай ущерб здоровью людей или банкротство как установленный факт.",
        "- 150–250 слов, живой газетный стиль, на русском языке.",
        "- Последней строкой напиши: «Учебная гипотеза модели последствий, не прогноз по реальной реке».",
        "",
        "Факты:",
        *[f"- {fact}" for fact in facts],
    ])


# --- consequences_model/explain.py: Шаг 4: какие правила сработали и на что они опираются.

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


# --- consequences_model/model.py: Прогноз по итогу игры: протокол → исполнение → состояние города → исход.

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


# --- consequences_model/langflow_component.py: Langflow-компонент «Модель последствий»: обёртка над прогнозом; в собранном файле модель встроена рядом.

class ConsequencesModel(Component):
    display_name = "Модель последствий"
    description = "К чему придут город N и участники через 10 лет после голосования по протоколу."
    icon = "landmark"

    inputs = [
        DataInput(name="game_result", display_name="Game Result", required=True,
                  info="Выход оркестратора игры: взносы, условия протокола, голоса и профили игроков."),
    ]
    outputs = [
        Output(name="forecast", display_name="Прогноз", method="build_forecast"),
        Output(name="report", display_name="Отчёт", method="build_report"),
        Output(name="herald_prompt", display_name="Промпт «Вестника»", method="build_herald_prompt"),
    ]

    def _forecast(self) -> dict[str, Any]:
        # Прогноз дешёвый и детерминированный: считаем на каждый выход, без кэша между запусками.
        return forecast(self.game_result.data)

    def build_forecast(self) -> Data:
        return Data(data=self._forecast())

    def build_report(self) -> Message:
        return Message(text=self._forecast()["report"])

    def build_herald_prompt(self) -> Message:
        return Message(text=self._forecast()["herald_prompt"])
