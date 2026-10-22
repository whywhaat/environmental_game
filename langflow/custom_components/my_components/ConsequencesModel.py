"""Модель последствий для Langflow. СОБРАНО из пакета consequences_model — руками не править.

Пересобрать: python -m consequences_model.build_component
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from dataclasses import asdict
from lfx.custom import Component
from lfx.io import DataInput, IntInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message


# --- consequences_model/catalog.py: Данные модели: роли, компоненты сметы, уровни, пороги и каталог исходов.

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

WATER = "water_supply"
MONITORING = "independent_monitoring"
RECONSTRUCTION = "reconstruction"

COMPONENT_LABELS = {
    WATER: "Вода и заморозка тарифов",
    MONITORING: "Независимый мониторинг",
    RECONSTRUCTION: "Реконструкция",
}

# Голос по протоколу: None — воздержался (ядро пишет None, если участник не проголосовал).
VOTE_LABELS = {True: "за", False: "против", None: "воздержался"}

# Исполнение обязательства.
FULL = "полностью"
PARTIAL = "частично"
NONE = "нет"

# Пороги — наша формализация; калибруются после тестового прогона.
PAID_PART = {FULL: 1.0, PARTIAL: 0.5, NONE: 0.0}  # какая часть взноса реально внесена
MIN_COVERAGE_TO_BUILD = 0.75
DELAY_PARTIAL_RECONSTRUCTION = 12  # мес.
DELAY_UNDERFUNDED = 24  # мес.
DEFAULT_PROMISED_MONTHS = 36
HORIZON_MONTHS = 120

# Учебник, табл. 4: независимая лаборатория, а не «своя».
INDEPENDENT_MONITORS = (REGION, NGO)
RAISE = {NONE: PARTIAL, PARTIAL: FULL, FULL: FULL}

# Каталог исходов: название, условие словами (для объяснения) и описание для отчёта и «Вестника».
OUTCOMES = {
    "nedostroy": {
        "name": "Недострой",
        "rule": "очистные не введены к году 10",
        "description": "Через 10 лет очистные так и не введены: реконструкцию недофинансировали, выполнили не "
                       "полностью или за неё никто не отвечал. Река остаётся загрязнённой, временные меры к этому "
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
        "rule": "очистные введены позже обещанного срока, а вода и заморозка тарифов не обеспечены полностью",
        "description": "Очистные введены позже обещанного, а подвоз воды и заморозка тарифов на время работ "
                       "были обеспечены не полностью. Годы ожидания с перебоями в воде подорвали доверие к "
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
        "rule": "очистные введены в обещанный срок, все три компонента выполнены полностью, мониторинг "
                "независимый, жители и НКО голосовали за",
        "description": "Очистные введены в обещанный срок, а независимый мониторинг с открытой публикацией "
                       "показывает всем состояние реки. Затраты разделены, жители и экологи участвовали в "
                       "решении, негатив к холдингу снижается, протестов нет. Это решение близко к эталонному "
                       "(учебник, табл. 7).",
    },
    "closed_data": {
        "name": "Чистая вода, закрытые данные",
        "rule": "мониторинг не независимый или выполнен не полностью",
        "description": "Очистные работают, но полноценного независимого мониторинга нет: замеры ведёт "
                       "заинтересованная сторона, ведутся не полностью или не ведутся вовсе, и истинное состояние "
                       "реки неизвестно (учебник, табл. 4). Жители не верят отчётам, недоверие держится.",
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

# Методичка: у жителей и НКО нет денег и полномочий строить и возить воду; экспертиза и контроль — область НКО.
WITHOUT_MEANS = (RESIDENTS, NGO)

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
WATER_MAX_MONTHS = 48  # подвоз воды и заморозка тарифов — до ввода, но не дольше 4 лет
LATE_WITHOUT_WATER_MONTHS = 36
MEDIA_HOLDING_MIN_SHARE = 0.30

FUNDING_FROM_OTHERS_SHARE = 0.70  # администрация получила деньги от области и холдинга
REGION_LOW_COST_SHARE = 0.20

# Источники правил для объяснения.
SOURCE_FORMALIZATION = "наша формализация"
SOURCE_TABLE_4 = "Учебник, 3.1, табл. 4"
SOURCE_TABLE_4_ANALOGY = "наша формализация по аналогии с учебником, 3.1, табл. 4"
SOURCE_TABLE_5_LAW = "Учебник, 3.1, табл. 5 («Право»)"
SOURCE_TABLE_5_ECONOMY = "Учебник, 3.1, табл. 5 («Экономика»)"
SOURCE_METODICHKA = "Методичка"
SOURCE_OUTCOMES = "каталог исходов (наша формализация)"

SOURCE_TABLES_4_5 = "Учебник, 3.1, табл. 4, 5"
SOURCE_TABLES_5_7 = "Учебник, 3.1, табл. 5, 7"
SOURCE_TABLE_5_POLITICS = "Учебник, 3.1, табл. 5 («Политика»); методичка"

# Почему подсистема получила уровень: критерий словами и опора.
SUBSYSTEM_CRITERIA = {
    "ecology": (SOURCE_TABLES_4_5, {
        ACHIEVED: "очистные введены, мониторинг независимый и выполнен полностью",
        PARTLY: "очистные введены без полноценного независимого мониторинга или жителям пока возят воду",
        FAILED: "нет ни очистных, ни подвоза воды"}),
    "media": (SOURCE_TABLES_5_7, {
        ACHIEVED: "холдинг полностью внёс заметную долю (от 30%), а независимый мониторинг публикует данные",
        PARTLY: "есть только одно из двух: заметный вклад холдинга или независимые открытые данные",
        FAILED: "нет ни заметного вклада холдинга, ни независимых открытых данных"}),
    "economy": (SOURCE_TABLE_5_ECONOMY, {
        ACHIEVED: "нагрузка не перекошена: холдинг меньше 70%, администрация меньше 50%, кредитов нет",
        PARTLY: "нагрузка смещена на холдинг или город либо кто-то платит в кредит",
        FAILED: "холдинг платит в кредит"}),
    "law": (SOURCE_TABLE_5_LAW, {
        ACHIEVED: "независимый мониторинг выполнен, у каждого компонента есть ответственный",
        PARTLY: "мониторинг есть, но не независимый или неполный либо не у всех компонентов есть ответственный",
        FAILED: "мониторинг не выполняется или за него никто не отвечает"}),
    "politics": (SOURCE_TABLE_5_POLITICS, {
        ACHIEVED: "жители голосовали «за», и вода у них есть",
        PARTLY: "жители не поддержали протокол или остались без надёжной воды, но до массовых протестов не дошло",
        FAILED: "жители и НКО не поддержали протокол, или очистных нет, или жители долго без воды"}),
}


# --- consequences_model/protocol.py: Адаптер: итог игры ядра переговоров → Протокол.

@dataclass(frozen=True)
class Protocol:
    total_cost: int
    contributions: dict[str, int]
    responsible: dict[str, str | None]
    votes: dict[str, bool | None]
    accepted: bool
    budgets: dict[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default=())

    def supports(self, role: str) -> bool:
        """Роль поддержала протокол: голос «за». «Против» и «воздержался» — не поддержка."""
        return self.votes[role] is True

    @property
    def monitoring_is_independent(self) -> bool:
        """Учебник, табл. 4: независимая лаборатория, а не «своя»."""
        return self.responsible[MONITORING] in INDEPENDENT_MONITORS

    def lacks_means(self, component: str) -> bool:
        """Методичка: у жителей и НКО нет денег и полномочий строить и возить воду; контроль — область НКО."""
        owner = self.responsible[component]
        return owner in WITHOUT_MEANS and not (component == MONITORING and owner == NGO)

    @property
    def credit(self) -> list[str]:
        """Роли, чей взнос больше бюджета: превышение покрыто кредитом."""
        return [role for role, amount in self.contributions.items()
                if role in self.budgets and amount > self.budgets[role]]


def read_protocol(game_result: dict[str, Any], players: list[dict[str, Any]] | None = None) -> Protocol:
    """Принимает Data.data оркестратора ({"result": ...}) или голый result.json.

    players — профили игроков с бюджетами; без них правила о кредите не срабатывают.
    """
    result = game_result["result"] if "result" in game_result else game_result
    fields = result["final_state"]["fields"]
    warnings: list[str] = []

    def value(path: str) -> Any:
        records = fields.get(path) or []
        distinct = {json.dumps(record["value"], sort_keys=True) for record in records}
        if len(distinct) > 1:
            warnings.append(f"Конфликт в поле {path}: обязательство считается незаявленным.")
            return None
        return records[0]["value"] if records else None

    contributions = {}
    for role in ROLE_LABELS:
        amount = value(f"participants.{role}.contribution")
        if amount:
            contributions[role] = amount
    responsible = {}
    for component in COMPONENT_LABELS:
        path = f"components.{component}.responsible"
        owner = value(path)
        if owner is not None and owner not in ROLE_LABELS:
            warnings.append(f"Неизвестная роль {owner} в поле {path}: компонент без ответственного.")
            owner = None
        responsible[component] = owner
    for path in fields:
        kind, ident = path.split(".")[:2]
        if kind == "components" and ident not in COMPONENT_LABELS:
            warnings.append(f"Неизвестный компонент {ident}: в прогнозе не учитывается.")
        if kind == "participants" and ident not in ROLE_LABELS:
            warnings.append(f"Неизвестная роль {ident} в поле {path}: взнос не учитывается.")
    # True — «за», False — «против», None — воздержался (ядро пишет None, если участник не проголосовал).
    raw_votes = result.get("votes") or {}
    votes = {role: raw_votes.get(role) if raw_votes.get(role) in (True, False) else None for role in ROLE_LABELS}
    budgets = {player["id"]: player["budget"] for player in players or [] if player.get("id") in ROLE_LABELS}
    return Protocol(result["total_cost"], contributions, responsible, votes,
                    bool(result.get("accepted")), budgets, tuple(warnings))


# --- consequences_model/state.py: Шаг 2: состояние города — уровни пяти подсистем учебника в момент прогноза.

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
    support, shares, credit = facts["support"], facts["shares"], facts["credit"]
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
        "politics": level((not support[RESIDENTS] and not support[NGO]) or late_without_water or (horizon and not built),
                          support[RESIDENTS] and residents_have_water),
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


def temporary_measures(result: dict[str, Any]) -> str:
    level = result["execution"]["components"][WATER]
    owner = result["protocol"]["responsible"][WATER]
    who = f" (отвечает: {ROLE_LABELS[owner]})" if owner else ""
    return f"подвоз воды и заморозка тарифов выполнены: {level}{who}; действуют до ввода очистных, " \
           f"не дольше {WATER_MAX_MONTHS // 12} лет"


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
    lines += ["", "| Компонент сметы | Ответственный | Исполнение |", "| --- | --- | --- |"]
    for component, label in COMPONENT_LABELS.items():
        owner = protocol["responsible"][component]
        lines.append(f"| {label} | {ROLE_LABELS[owner] if owner else '—'} | {execution['components'][component]} |")

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
              "Пороги правил и коэффициенты исполнения — наша формализация, требуют согласования с заказчицей. "
              f"Обещанный срок ввода очистных — {result['promised_months']} мес., в игре он не обсуждается. "
              "Прогноз — учебная гипотеза, а не прогноз по реальной реке."]
    if result["warnings"]:
        lines += ["", "## Предупреждения", ""] + [f"- {warning}" for warning in result["warnings"]]
    return "\n".join(lines)


def render_herald_prompt(result: dict[str, Any]) -> str:
    outcome, state = result["outcome"], result["state"]
    year_10 = state["year_10"]["subsystems"]
    facts = [f"Исход: «{outcome['name']}». {outcome['description']}",
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


# --- consequences_model/model.py: Прогноз по итогу игры: протокол → исполнение → состояние города → исход.

def execute(protocol: Protocol, excluded: str | None = None) -> dict[str, Any]:
    """Шаг 1: насколько каждая роль выполнила свои обязательства.

    excluded — роль, чьи обязательства считаются невыполненными (контрфактический прогон).
    """
    def by_vote(owner: str | None) -> str:
        if owner is None or owner == excluded:
            return NONE
        if protocol.accepted:
            return FULL if protocol.supports(owner) else PARTIAL
        return PARTIAL if protocol.supports(owner) else NONE

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
    ("without_residents", lambda f: not f["support"][RESIDENTS] and not f["support"][NGO]),
    ("dolgostroy", lambda f: f["month"] > f["promised_months"] and f["components"][WATER] != FULL),
    ("holding_paid", lambda f: f["shares"][HOLDING] >= HOLDING_PAID_ALL_SHARE),
    ("city_paid", lambda f: f["shares"][ADMIN] >= CITY_PAID_ALL_SHARE),
    ("open_river", lambda f: f["month"] == f["promised_months"] and f["monitoring_ok"]
        and all(value == FULL for value in f["components"].values())
        and f["support"][RESIDENTS] and f["support"][NGO]),
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
        "support": {role: protocol.supports(role) for role in ROLE_LABELS},
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


# --- consequences_model/langflow_component.py: Langflow-компонент «Модель последствий»: обёртка над прогнозом; в собранном файле модель встроена рядом.

class ConsequencesModel(Component):
    display_name = "Модель последствий"
    description = "К чему придут город N и участники через 10 лет после голосования по протоколу."
    icon = "landmark"

    inputs = [
        DataInput(name="game_result", display_name="Game Result", required=True,
                  info="Выход оркестратора игры."),
        DataInput(name="players", display_name="Players", is_list=True, required=False,
                  info="Профили игроков: из них берутся бюджеты, чтобы видеть кредиты."),
        IntInput(name="promised_months", display_name="Обещанный срок ввода очистных, мес.",
                 value=DEFAULT_PROMISED_MONTHS, info="В игре срок не обсуждается; это допущение модели."),
    ]
    outputs = [
        Output(name="forecast", display_name="Прогноз", method="build_forecast"),
        Output(name="report", display_name="Отчёт", method="build_report"),
        Output(name="herald_prompt", display_name="Промпт «Вестника»", method="build_herald_prompt"),
    ]

    def _forecast(self) -> dict[str, Any]:
        # Прогноз дешёвый и детерминированный: считаем на каждый выход, без кэша между запусками.
        players = [item.data for item in (self.players or [])]
        return forecast(self.game_result.data, players, int(self.promised_months))

    def build_forecast(self) -> Data:
        return Data(data=self._forecast())

    def build_report(self) -> Message:
        return Message(text=self._forecast()["report"])

    def build_herald_prompt(self) -> Message:
        return Message(text=self._forecast()["herald_prompt"])
