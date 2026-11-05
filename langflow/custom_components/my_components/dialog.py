"""СОБРАНО из negotiation_game — руками не править.

Пересобрать компонент и код в обоих flow: python -m negotiation_game.build_component
"""


# --- negotiation_game/schemas.py ---

"""Структурированные ответы игроков; не зависят от Langflow и LangChain."""

import copy
import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt,
    StrictStr, model_validator,
)

PlayerId = Annotated[StrictStr, Field(min_length=1)]
DemandId = Annotated[StrictStr, Field(min_length=1)]
TurnText = Annotated[StrictStr, Field(min_length=1)]
AmountMillions = Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]
LeverValue = StrictBool | StrictInt | TurnText
DemandValue = LeverValue | StrictFloat
Vote = Literal["for", "against", "abstain"]
DemandDecision = Literal["accept", "reject", "counter"]


class TurnSchema(BaseModel):
    """Общие ограничения структурированного ответа игрока (Pydantic v2)."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


class Demand(TurnSchema):
    """Новое требование; его ID назначает Python после проверки хода."""

    target: PlayerId = Field(description="Стабильный player_id адресата требования")
    demand_type: Literal["pledge", "lever"]
    lever_id: TurnText | None = Field(
        description="ID условия протокола для lever; null для pledge"
    )
    value: DemandValue = Field(
        description="Для pledge — сумма в млн рублей; для lever — значение условия"
    )
    text: TurnText = Field(description="Краткая суть требования")
    counter_to: DemandId | None = Field(
        description="ID исходного требования для встречного предложения; иначе null"
    )

    @model_validator(mode="after")
    def validate_demand(self) -> "Demand":
        if self.demand_type == "pledge":
            if self.lever_id is not None:
                raise ValueError("Для pledge lever_id должен быть null")
            # bool является подклассом int, но денежной суммой не является.
            if type(self.value) not in (int, float) or self.value < 0:
                raise ValueError("Для pledge value — неотрицательное число миллионов")
        else:
            if self.lever_id is None:
                raise ValueError("Для lever требуется lever_id")
            if type(self.value) not in (bool, int, str):
                raise ValueError("Для lever value должен иметь тип bool, int или str")
        return self


class DemandResponse(TurnSchema):
    """Решение адресата по конкретному требованию; обязательства не меняет."""

    demand_id: DemandId
    decision: DemandDecision


class PlayerTurn(TurnSchema):
    """Ответ игрока. Все поля обязательны, отсутствие действия задаётся явно.

    Существование ID, права игрока, соответствие этапу и согласованность действий
    требуют отдельной проверки по состоянию игры до атомарного применения хода.
    """

    speech: TurnText = Field(description="Публичная речь игрока без технических команд")
    own_contribution_millions: AmountMillions | None = Field(
        description="Новый собственный взнос в млн рублей; null — без изменений; 0 — обнулить"
    )
    own_levers: dict[TurnText, LeverValue] = Field(
        description="Изменения собственных условий по их ID; пустой объект — без изменений"
    )
    demands: list[Demand] = Field(description="Новые требования и встречные предложения")
    responses: list[DemandResponse] = Field(description="Ответы на адресованные игроку требования")
    withdrawn_demands: list[DemandId] = Field(description="ID отзываемых собственных требований")
    addressed_to: PlayerId | None = Field(
        description="player_id адресата публичной реплики; null — обращение ко всем"
    )
    vote: Vote | None = Field(
        description="Голос по итоговому протоколу; вне финального голосования — null"
    )


class ContributionInput(TurnSchema):
    action: Literal["unchanged", "set"]
    amount_millions: AmountMillions

    @model_validator(mode="after")
    def validate_unchanged(self) -> "ContributionInput":
        if self.action == "unchanged" and self.amount_millions != 0:
            raise ValueError("При action=unchanged amount_millions должен быть 0 (служебное значение)")
        return self


class ConditionInput(TurnSchema):
    lever_id: TurnText
    value_json: TurnText = Field(
        description='Значение условия строкой: true/false, 36, 70% или holding. Строка должна точно соответствовать допустимому значению условия.'
    )


class PledgeDemandInput(TurnSchema):
    target: PlayerId
    amount_millions: AmountMillions
    text: TurnText
    counter_to: StrictStr = Field(description="Точный ID встречного требования; пустая строка для нового")


class ConditionDemandInput(ConditionInput):
    target: PlayerId
    text: TurnText
    counter_to: StrictStr = Field(description="Точный ID встречного требования; пустая строка для нового")


class PlayerTurnInput(TurnSchema):
    """Формат модели без union и null: Gpt2Giga обрезает альтернативы типов.

    Это транспортный DTO, а не замена внутреннего игрового контракта.
    """

    speech: TurnText
    own_contribution: ContributionInput
    own_conditions: list[ConditionInput]
    pledge_demands: list[PledgeDemandInput]
    condition_demands: list[ConditionDemandInput]
    responses: list[DemandResponse]
    withdrawn_demands: list[DemandId]
    addressed_to: TurnText = Field(description="Точный player_id; everyone — обращение ко всем")
    vote: Literal["none", "for", "against", "abstain"]


def player_turn_schema() -> type[PlayerTurn]:
    """Разрешает аннотации в пространстве исполнения динамического компонента.

    Langflow компилирует классы с отложенными аннотациями, а их __module__ может
    указывать на загрузчик, где нет TurnText и других локальных типов. Поэтому
    передаём фактическое пространство исполнения явно и собираем вложенные модели
    до PlayerTurn. Вызов нужен при использовании схемы: загрузчик Langflow может
    пропускать выражения верхнего уровня, оставляя только определения.
    """
    namespace = globals()
    for schema in (Demand, DemandResponse, PlayerTurn, ContributionInput, ConditionInput,
                   PledgeDemandInput, ConditionDemandInput, PlayerTurnInput):
        schema.model_rebuild(_types_namespace=namespace)
    return PlayerTurn


def player_turn_tool_schema(player_ids: dict, voting: bool = False,
                            levers_config: list[dict] | None = None) -> dict:
    """Транспортная схема; внутренний PlayerTurn проверяется после нормализации.

    Два варианта требования разделены для генератора. Игровые проверки остаются
    обязательными: JSON Schema не выражает права владельцев и состояние игры.
    """
    player_turn_schema()
    schema = copy.deepcopy(PlayerTurnInput.model_json_schema())
    schema["title"] = "PlayerTurn"
    ids = list(player_ids)
    if not ids:
        raise ValueError("Нет допустимых player_id")
    definitions = schema["$defs"]
    for title in ("PledgeDemandInput", "ConditionDemandInput"):
        definitions[title]["properties"]["target"] = {
            "type": "string", "enum": ids, "description": "Точный player_id адресата требования"}
    if levers_config:
        lever_ids = [lever["id"] for lever in levers_config]
        for title in ("ConditionInput", "ConditionDemandInput"):
            definitions[title]["properties"]["lever_id"]["enum"] = lever_ids
    fields = schema["properties"]
    fields["addressed_to"] = {"type": "string", "enum": [*ids, "everyone"],
                              "description": "Точный player_id; обращение ко всем — everyone"}
    if voting:
        definitions["ContributionInput"]["properties"]["action"]["enum"] = ["unchanged"]
        definitions["ContributionInput"]["properties"]["amount_millions"]["maximum"] = 0
        fields["addressed_to"]["enum"] = ["everyone"]
        for name in ("own_conditions", "pledge_demands", "condition_demands", "responses", "withdrawn_demands"):
            fields[name]["maxItems"] = 0
        fields["vote"] = {"type": "string", "enum": ["for", "against", "abstain"]}
    else:
        fields["vote"] = {"type": "string", "enum": ["none"]}
    return schema


def decode_player_turn_input(payload: dict, levers_config: list[dict] | None = None) -> dict:
    """Детерминированный перевод явно заданных действий во внутренний контракт."""
    player_turn_schema()
    wire = PlayerTurnInput.model_validate(payload)

    conditions = {lever["id"]: lever for lever in (levers_config or [])}

    def value_json(text, path, lever_id):
        condition = conditions.get(lever_id, {})
        # Это явный выбор из enum строкового условия, без дополнительного
        # JSON-кодирования. Произвольные строки и значения других типов не угадываем.
        if condition.get("type") == "str" and text in condition.get("allowed", []):
            return text
        try:
            return json.loads(text)
        except ValueError as exc:
            allowed = condition.get("allowed")
            hint = f"; допустимые строки: {allowed}" if condition.get("type") == "str" else ""
            raise ValueError(f"{path}.value_json: неверное значение условия {lever_id}{hint}") from exc

    own_levers = {}
    for index, condition in enumerate(wire.own_conditions):
        if condition.lever_id in own_levers:
            raise ValueError(f"own_conditions.{index}.lever_id: повторное изменение одного условия")
        own_levers[condition.lever_id] = value_json(condition.value_json, f"own_conditions.{index}", condition.lever_id)
    demands = [{"target": demand.target, "demand_type": "pledge", "lever_id": None,
                "value": demand.amount_millions, "text": demand.text,
                "counter_to": demand.counter_to or None} for demand in wire.pledge_demands]
    demands.extend({"target": demand.target, "demand_type": "lever", "lever_id": demand.lever_id,
                    "value": value_json(demand.value_json, f"condition_demands.{index}", demand.lever_id),
                    "text": demand.text, "counter_to": demand.counter_to or None}
                   for index, demand in enumerate(wire.condition_demands))
    return {"speech": wire.speech,
            "own_contribution_millions": (wire.own_contribution.amount_millions
                                           if wire.own_contribution.action == "set" else None),
            "own_levers": own_levers, "demands": demands,
            "responses": [response.model_dump() for response in wire.responses],
            "withdrawn_demands": list(wire.withdrawn_demands),
            "addressed_to": None if wire.addressed_to == "everyone" else wire.addressed_to,
            "vote": None if wire.vote == "none" else wire.vote}


def normalize_player_turn(payload: dict, player_ids: dict,
                          audit: list[dict] | None = None,
                          levers_config: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Только однозначные преобразования; суммы и смысл действий не исправляются."""
    if not isinstance(payload, dict):
        raise ValueError("PlayerTurn должен быть JSON-объектом")
    normalized = copy.deepcopy(payload)
    changes = [] if audit is None else audit
    aliases = {}
    for player_id, name in player_ids.items():
        aliases.setdefault(name, []).append(player_id)

    def replace(record, key, value, path):
        before = record[key]
        record[key] = value
        changes.append({"field": path, "before": before, "after": value})

    def nullable(record, key, path):
        if key in record and isinstance(record[key], str) and record[key].strip() in ("", "null"):
            replace(record, key, None, path)

    def player_reference(record, key, path, everyone=False):
        value = record.get(key)
        if not isinstance(value, str) or value in player_ids:
            return
        if everyone and value == "все":
            replace(record, key, None, path)
        elif len(aliases.get(value, [])) == 1:
            replace(record, key, aliases[value][0], path)

    if any(key in normalized for key in ("own_contribution", "pledge_demands", "condition_demands", "own_conditions")):
        for key, sentinel in (("addressed_to", "everyone"), ("vote", "none")):
            if key in normalized:
                value = normalized[key]
                if value is None or (isinstance(value, str) and value.strip() in ("", "null")):
                    replace(normalized, key, sentinel, key)
                elif key == "addressed_to" and value == "все":
                    replace(normalized, key, sentinel, key)
        for key in ("pledge_demands", "condition_demands"):
            if not isinstance(normalized.get(key), list):
                continue
            for index, demand in enumerate(normalized[key]):
                if not isinstance(demand, dict) or "counter_to" not in demand:
                    continue
                value = demand["counter_to"]
                if value is None or (isinstance(value, str) and value.strip() == "null"):
                    replace(demand, "counter_to", "", f"{key}.{index}.counter_to")
        normalized = decode_player_turn_input(normalized, levers_config)
        changes.append({"field": "$", "operation": "decode_player_turn_input"})

    for key in ("own_contribution_millions", "addressed_to", "vote"):
        nullable(normalized, key, key)
    player_reference(normalized, "addressed_to", "addressed_to", everyone=True)
    demands = normalized.get("demands")
    if isinstance(demands, list):
        for index, demand in enumerate(demands):
            if not isinstance(demand, dict):
                continue
            prefix = f"demands.{index}."
            for key in ("lever_id", "counter_to"):
                nullable(demand, key, prefix + key)
            player_reference(demand, "target", prefix + "target")
    return normalized, changes


def player_turn_examples(player_ids: dict, levers_config: list[dict], voting: bool,
                         speaker: str | None = None) -> str:
    """Полные образцы синтаксиса, не готовые решения за участника."""
    empty = {"speech": "Публичная позиция участника.",
             "own_contribution": {"action": "unchanged", "amount_millions": 0},
             "own_conditions": [], "pledge_demands": [], "condition_demands": [],
             "responses": [], "withdrawn_demands": [],
             "addressed_to": "everyone", "vote": "abstain" if voting else "none"}
    examples = {"голосование" if voting else "без действий": empty}
    if not voting:
        target = next((pid for pid, name in player_ids.items() if name != speaker), None)
        if target is not None:
            pledge = copy.deepcopy(empty)
            pledge["pledge_demands"] = [{"target": target, "amount_millions": 10,
                                         "text": "Просим взнос 10 млн рублей.", "counter_to": ""}]
            examples["денежное требование"] = pledge
        # Используем реальное условие и его владельца только как пример формы.
        for lever in levers_config:
            owner_id = next((pid for pid, name in player_ids.items() if name == lever.get("owner")), None)
            if owner_id and player_ids[owner_id] != speaker and lever.get("type") == "bool":
                condition = copy.deepcopy(empty)
                condition["condition_demands"] = [{"target": owner_id, "lever_id": lever["id"],
                    "value_json": "true", "text": "Просим включить условие протокола.", "counter_to": ""}]
                examples["требование условия"] = condition
                break
    return "ПРИМЕРЫ ФОРМАТА (не копируй решения и суммы): " + json.dumps(examples, ensure_ascii=False)


# --- negotiation_game/rules.py ---

"""Проверка прав и атомарная подготовка хода без LLM и Langflow.

prepare_player_turn возвращает новое состояние и действие, не изменяя входное.
"""

import copy
import math



def is_owner(owner: str, speaker: str) -> bool:
    if not owner or not speaker:
        return False
    o = owner.lower().strip()
    s = speaker.lower().strip()
    if o in s or s in o:
        return True
    role_groups = [
        {"администр", "мэр", "мэрия", "город"},
        {"пром", "холдинг", "завод", "комбинат", "предприяти"},
        {"регион", "област", "губернатор", "правительств"},
        {"жител", "горожан", "активист", "населени"},
        {"эко", "нко", "эколог"},
    ]
    for group in role_groups:
        o_match = any(w in o for w in group)
        s_match = any(w in s for w in group)
        if o_match and s_match:
            return True
    return False


def prepare_player_turn(answer: PlayerTurn, speaker: str, state: dict,
                         levers_config: list[dict], total_cost: float,
                         voting: bool = False) -> tuple[dict, dict]:
    """Проверяет ход на копии. Исходное состояние не меняется даже при ошибке."""
    names_by_id = state["player_ids"]
    if speaker not in names_by_id.values():
        raise ValueError("Неизвестный игрок")
    if answer.addressed_to is not None and answer.addressed_to not in names_by_id:
        raise ValueError("addressed_to должен содержать известный player_id")
    target = names_by_id.get(answer.addressed_to)
    if voting:
        if answer.vote is None:
            raise ValueError("На голосовании требуется for, against или abstain")
        if (answer.own_contribution_millions is not None or answer.own_levers
                or answer.demands or answer.responses or answer.withdrawn_demands
                or answer.addressed_to is not None):
            raise ValueError("На голосовании разрешены только speech и vote")
        return copy.deepcopy(state), {
            "type": "VOTE", "vote": {"for": "ЗА", "against": "ПРОТИВ",
                                      "abstain": "ВОЗДЕРЖАЛСЯ"}[answer.vote],
            "comment": answer.speech, "message": answer.speech, "status": "ok",
        }
    if answer.vote is not None:
        raise ValueError("Вне голосования vote должен быть null")

    candidate = copy.deepcopy(state)
    applied = []
    conditions = {lev["id"]: lev for lev in levers_config}

    def money(value, player):
        rubles = value * 1_000_000.0
        if not math.isfinite(rubles):
            raise ValueError("Сумма слишком велика")
        budget = state["players"][player].get("max_budget")
        if budget is not None and rubles > budget:
            raise ValueError(f"Взнос {player} превышает бюджет")
        return rubles

    def condition_value(lever_id, value, owner):
        lev = conditions.get(lever_id)
        if lev is None:
            raise ValueError(f"Неизвестное условие {lever_id}")
        if not is_owner(lev["owner"], owner):
            raise ValueError(f"Условие {lever_id} не принадлежит {owner}")
        expected = {"bool": bool, "int": int, "str": str}.get(lev["type"])
        if expected is None or type(value) is not expected:
            raise ValueError(f"Неверный тип значения условия {lever_id}")
        if "allowed" in lev and value not in lev["allowed"]:
            raise ValueError(f"Недопустимое значение условия {lever_id}")
        if expected is int and not lev.get("min_value", value) <= value <= lev.get("max_value", value):
            raise ValueError(f"Значение {lever_id} вне допустимого диапазона")
        return lev

    if answer.own_contribution_millions is not None:
        value = money(answer.own_contribution_millions, speaker)
        candidate["players"][speaker].update(
            contribution=value, share=round(value / total_cost * 100, 2) if total_cost else 0.0)
        applied.append({"kind": "PLEDGE", "player": speaker, "contribution": value,
                        "share": candidate["players"][speaker]["share"]})
    for lever_id, value in answer.own_levers.items():
        lev = condition_value(lever_id, value, speaker)
        candidate["levers"][lever_id] = value
        applied.append({"kind": "LEVER", "lever_id": lever_id,
                        "title": lev["title"], "value": value})
    # Пересчитываем расходы по всем включённым собственным условиям.
    candidate["players"][speaker]["spent_on_levers"] = sum(
        float(lev.get("cost", 0)) for lev in levers_config
        if is_owner(lev["owner"], speaker)
        and (candidate["levers"].get(lev["id"]) is True
             or (type(candidate["levers"].get(lev["id"])) is int
                 and candidate["levers"][lev["id"]] > 0)))

    existing = {d["id"]: d for d in candidate["demands"]}
    response_ids = [r.demand_id for r in answer.responses]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("Нельзя дважды ответить на одно требование в одном ходе")
    if len(answer.withdrawn_demands) != len(set(answer.withdrawn_demands)):
        raise ValueError("Повторяющийся ID отзываемого требования")
    counters = {r.demand_id for r in answer.responses if r.decision == "counter"}
    counter_links = {d.counter_to for d in answer.demands if d.counter_to is not None}
    if counters != counter_links:
        raise ValueError("Каждый counter требует нового demands с соответствующим counter_to")

    for response in answer.responses:
        demand = existing.get(response.demand_id)
        if demand is None or demand.get("withdrawn"):
            raise ValueError(f"Требование {response.demand_id} не существует или отозвано")
        if demand["target"] != speaker:
            raise ValueError("Ответить на требование может только его адресат")
        if response.decision == "accept":
            if demand["demand_type"] == "pledge":
                if answer.own_contribution_millions is None or not math.isclose(
                        candidate["players"][speaker]["contribution"], demand["value"], abs_tol=1e-6):
                    raise ValueError("accept требует явно указать согласованный собственный взнос")
            elif (demand["lever_id"] not in answer.own_levers
                  or answer.own_levers[demand["lever_id"]] != demand["value"]):
                raise ValueError("accept требует явно указать согласованное собственное условие")
        demand["response"] = response.decision
        applied.append({"kind": "DEMAND_RESPONSE", "demand_id": response.demand_id,
                        "decision": response.decision})
    for demand_id in answer.withdrawn_demands:
        demand = existing.get(demand_id)
        if demand is None or demand["from"] != speaker or demand.get("withdrawn"):
            raise ValueError("Отозвать можно только существующее собственное требование")
        demand.update(withdrawn=True, status="withdrawn")
        applied.append({"kind": "DEMAND_WITHDRAWN", "demand_id": demand_id})

    for demand in answer.demands:
        if demand.target not in names_by_id:
            raise ValueError(f"Неизвестный player_id {demand.target}")
        recipient = names_by_id[demand.target]
        if recipient == speaker:
            raise ValueError("Нельзя адресовать требование самому себе")
        if demand.counter_to is not None and recipient != existing[demand.counter_to]["from"]:
            raise ValueError("Встречное предложение адресуется автору исходного требования")
        if demand.demand_type == "pledge":
            value = money(demand.value, recipient)
            title = "Взнос"
        else:
            lev = condition_value(demand.lever_id, demand.value, recipient)
            value, title = demand.value, lev["title"]
        # counter_to ссылается на проверенный выше ответ адресата.
        seq = candidate.get("next_demand_id", 1)
        demand_id = f"d{seq}"
        candidate["next_demand_id"] = seq + 1
        record = {"id": demand_id, "from": speaker, "target": recipient,
                  "target_id": demand.target, "demand_type": demand.demand_type,
                  "lever_id": demand.lever_id, "value": value, "text": demand.text,
                  "title": title, "counter_to": demand.counter_to, "status": "pending"}
        candidate["demands"].append(record)
        applied.append({"kind": "DEMAND", **copy.deepcopy(record)})

    # Статус согласования следует фактическим обязательствам, а не одной реплике.
    for demand in candidate["demands"]:
        if demand.get("withdrawn"):
            continue
        if demand["demand_type"] == "pledge":
            contribution = candidate["players"][demand["target"]]["contribution"]
            met = contribution is not None and contribution >= demand["value"]
        else:
            actual = candidate["levers"][demand["lever_id"]]
            met = actual <= demand["value"] if type(actual) is int else actual == demand["value"]
        demand["status"] = "agreed" if met else "pending"
    return candidate, {"type": "ACTION", "message": answer.speech,
                       "target": target, "applied": applied, "status": "ok"}


# --- negotiation_game/report.py ---

"""Текстовое представление игры: отчёт, протокол, история и сводки для промптов.

Функции читают переданные данные и возвращают строки; не вызывают модели и не
изменяют состояние. Модуль не зависит от Langflow или LangChain.
"""



def render_partial_report(history, state, total_cost, levers_config, status, error=None) -> str:
    """Промежуточный отчёт не объявляет протокол принятым или отклонённым."""
    labels = {"running": "Игра выполняется", "failed": "Игра прервана из-за ошибки",
              "cancelled": "Игра отменена"}
    lines = ["# Промежуточный отчёт игры", "", f"**Статус: {labels[status]}**",
             "Голосование не завершено. Итоговый результат не определён."]
    if error:
        lines.extend(["", f"Причина: {error}"])
    lines.extend(["", "## Зафиксированное состояние", render_state(state, levers_config, total_cost),
                  "", "## История выполненных ходов", render_history(history)])
    return "\n".join(lines)


def format_money(amount) -> str:
    if amount is None:
        return "—"
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return str(amount)
    if abs(val) >= 1_000_000:
        millions = val / 1_000_000
        if millions.is_integer():
            return f"{int(millions)} млн руб."
        return f"{millions:.1f} млн руб."
    if abs(val) > 0:
        return f"{int(val):,} руб.".replace(",", " ")
    return "0 руб."


def render_profile(profile, state, speaker):
    attrs = profile.get("attrs", {})
    base_budget = profile.get("money", 0)
    p_state = state.get("players", {}).get(speaker, {})
    spent_on_levers = p_state.get("spent_on_levers", 0.0)
    effective_budget = max(0.0, base_budget - spent_on_levers)

    lines = [
        f"Базовый бюджет стороны: {format_money(base_budget)}",
        f"Затраты на активированные обязательства: {format_money(spent_on_levers)}",
        f"Доступно на взнос в очистные сооружения: {format_money(effective_budget)}",
        f"Минимально приемлемая доля в проекте: {attrs.get('min_share', 0):g}%",
        f"Комфортная доля: {attrs.get('comfort_share', 0):g}%",
        f"Комфортный взнос: {format_money(attrs.get('comfort_money', 0))}",
        f"Красные линии стороны: {', '.join(attrs.get('red_lines', [])) if attrs.get('red_lines') else 'Нет жестких ограничений'}",
    ]
    return "\n".join(lines)


def render_player_levers(speaker: str, levers_config: list[dict], current_levers: dict) -> str:
    """Отображает нейтральную сводку полномочий конкретного игрока (без навязывания стратегии)."""
    owned = [lev for lev in levers_config if is_owner(lev["owner"], speaker)]
    if not owned:
        return "В вашем прямом ведении нет отдельных параметров (вы влияете на них через переговоры и вопросы)."

    lines = ["В ВАШЕМ ВЕДЕНИИ НАХОДЯТСЯ СЛЕДУЮЩИЕ ПАРАМЕТРЫ ПРОЕКТА:"]
    for lev in owned:
        curr_val = current_levers.get(lev["id"], lev.get("default"))
        cost_info = (
            f"Стоимость включения: {format_money(lev['cost'])} (списывается из вашего резерва, не поступает в общий фонд очистных)."
            if lev.get("cost")
            else "Без прямого списания бюджета."
        )
        lines.append(f"• [{lev['title']}]")
        lines.append(f"  - Текущее состояние: {curr_val}")
        lines.append(f"  - Экономика действия: {cost_info}")
        lines.append(f"  - lever_id для own_conditions: {lev['id']}; тип: {lev['type']}")
        if "allowed" in lev:
            lines.append(f"  - Допустимые значения: {lev['allowed']}")
        if "min_value" in lev or "max_value" in lev:
            lines.append(f"  - Диапазон: {lev.get('min_value')} — {lev.get('max_value')}")
        lines.append("  - Решение об изменении и условиях вы принимаете самостоятельно на основе своей стратегии.")
    return "\n".join(lines)


def render_calculator(state: dict, total_cost: float) -> str:
    total_c = 0.0
    total_s = 0.0
    player_rows = []

    for name, p_data in state["players"].items():
        c = p_data.get("contribution") or 0.0
        s = p_data.get("share") or 0.0
        total_c += c
        total_s += s
        spent = p_data.get("spent_on_levers", 0.0)
        spent_str = f" [доп. расход: {format_money(spent)}]" if spent > 0 else ""
        c_str = format_money(c) if c > 0 else "0 руб."
        player_rows.append(f"│ {name:<33} │ {c_str:<18} │ {s:>5.1f}% │{spent_str}")

    deficit = max(0.0, total_cost - total_c)
    if deficit > 1000:
        deficit_status = f"⚠️ ДЕФИЦИТ: {format_money(deficit)} (НЕ ЗАКРЫТ!)"
    else:
        deficit_status = "✅ ДЕФИЦИТ: 0 руб. (100% ЗАКРЫТ!)"

    lines = [
        "🧮 ОФИЦИАЛЬНЫЙ КАЛЬКУЛЯТОР БЮДЖЕТА ПРОЕКТА (ЕДИНЫЙ ИСТОЧНИК ИСТИНЫ):",
        "┌───────────────────────────────────┬────────────────────┬────────┐",
        "│ Участник переговоров              │ Официальный взнос  │ Доля   │",
        "├───────────────────────────────────┼────────────────────┼────────┤",
    ]
    lines.extend(player_rows)
    lines.extend([
        "├───────────────────────────────────┴────────────────────┴────────┤",
        f"│ ТРЕБУЕМЫЙ ЦЕЛЕВОЙ БЮДЖЕТ:   {format_money(total_cost):<35} │",
        f"│ СУММАРНО ОФИЦИАЛЬНО СОБРАНО: {format_money(total_c):<35} │",
        f"│ СТАТУС БЮДЖЕТА:             {deficit_status:<35} │",
        "└─────────────────────────────────────────────────────────────────┘",
        "⚡ ПРАВИЛА КАЛЬКУЛЯТОРА ДЛЯ ВСЕХ УЧАСТНИКОВ:",
        "1. Юридическую силу имеют ТОЛЬКО взносы из таблицы выше. Речь не меняет обязательства: новый взнос задаётся через own_contribution.action=set и own_contribution.amount_millions (млн рублей).",
        f"2. НЕ СЧИТАЙ В УМЕ! Реальный дефицит проекта прямо сейчас равен РОВНО {format_money(deficit)}.",
        "3. Категорически запрещено заявлять о закрытии дефицита или выходе на целевой бюджет, пока в строке СТАТУС БЮДЖЕТА не появится '100% ЗАКРЫТ'.",
    ])
    return "\n".join(lines)


def render_state(state: dict, levers_config: list[dict], total_cost: float) -> str:
    """Отрисовывает публичный Стол переговоров (калькулятор бюджета + доска условий + требования) для всех игроков."""
    calc_block = render_calculator(state, total_cost)

    lines = [calc_block, "\n📋 ДОСКА ДОГОВОРЕННОСТЕЙ И НЕФИНАНСОВЫХ КРУТИЛОК:"]
    current_levers = state.get("levers", {})
    for lev in levers_config:
        lid = lev["id"]
        val = current_levers.get(lid, lev.get("default"))
        owner = lev["owner"]
        cost_note = f" (списание с бюджета стороны: {format_money(lev['cost'])})" if lev.get("cost") else ""

        if lev["type"] == "bool":
            status_mark = "✅ ВКЛЮЧЕНО" if val else "❌ НЕ ВКЛЮЧЕНО"
            lines.append(f"  • [{status_mark}] {lev['title']} (В ведении: @{owner}{cost_note})")
        else:
            unit = " мес." if lev.get("type") == "int" else ""
            lines.append(f"  • ⏱️ Значение: {val}{unit} | {lev['title']} (В ведении: @{owner}{cost_note})")

    demands = state.get("demands", [])
    if demands:
        lines.append("\n⚡ АКТИВНЫЕ ТРЕБОВАНИЯ И ВСТРЕЧНЫЕ УСЛОВИЯ СТОРОН:")
        for d in demands:
            status_icon = {"agreed": "✅ СОГЛАСОВАНО", "withdrawn": "ОТОЗВАНО"}.get(d.get("status"), "⏳ НА РАССМОТРЕНИИ")
            if d.get("demand_type") == "pledge":
                lines.append(f"  • [{status_icon}] От @{d['from']} к @{d['target']}: Взнос {format_money(d['value'])}")
            else:
                v_str = "ВКЛ" if d.get("value") is True else ("ВЫКЛ" if d.get("value") is False else str(d.get("value")))
                lines.append(f"  • [{status_icon}] От @{d['from']} к @{d['target']}: {d.get('title')} -> {v_str}")
    else:
        lines.append("\n⚡ АКТИВНЫЕ ТРЕБОВАНИЯ И ВСТРЕЧНЫЕ УСЛОВИЯ: Нет зафиксированных встречных требований.")

    return "\n".join(lines)


def render_draft_treaty(state: dict, levers_config: list[dict], total_cost: float) -> str:
    """Формирует официальный текст Проекта Соглашения для голосования в чистом Markdown."""
    total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
    deficit = max(0.0, total_cost - total_c)
    total_s = sum(v["share"] for v in state["players"].values() if v.get("share") is not None)

    lines = [
        "> **Официальный проект соглашения**, сформированный сторонами для поименного голосования.\n",
        "### 1. Финансовый план реконструкции очистных сооружений",
        "| Сторона | Согласованный взнос | Доля в проекте |",
        "| :--- | :--- | :--- |",
    ]
    for name, p_data in state["players"].items():
        c = p_data.get("contribution") or 0.0
        s = p_data.get("share") or 0.0
        c_str = format_money(c)
        s_str = f"{s:g}%" if s > 0 else "0%"
        lines.append(f"| **{name}** | {c_str} | {s_str} |")

    lines.append("")
    lines.append(f"* **Суммарно подтверждено:** {format_money(total_c)} из {format_money(total_cost)} ({total_s:g}%)")
    if deficit > 1000:
        lines.append(f"* ⚠️ **Статус финансирования:** ДЕФИЦИТ {format_money(deficit)} (Остаточный дефицит бюджета)")
    else:
        lines.append("* ✅ **Статус финансирования:** БЮДЖЕТ ПОЛНОСТЬЮ ЗАКРЫТ (100% средств обеспечено)")

    lines.append("\n### 2. Пакет обязательств и условий реализации")
    current_levers = state.get("levers", {})
    for lev in levers_config:
        lid = lev["id"]
        val = current_levers.get(lid, lev.get("default"))
        owner = lev["owner"]
        desc = lev.get("description", "")
        cost_info = f" *Затраты стороны @{owner}: {format_money(lev['cost'])}.*" if lev.get("cost") else ""

        if lev["type"] == "bool":
            if val:
                lines.append(f"* ✅ **[УТВЕРЖДЕНО] {lev['title']}**")
                lines.append(f"  * Гарант исполнения: @{owner}.{cost_info}")
                if desc:
                    lines.append(f"  * Содержание: {desc}.")
            else:
                lines.append(f"* ❌ **[НЕ ВКЛЮЧЕНО] {lev['title']}**")
                lines.append(f"  * Сторона @{owner} не предоставила гарантий по данному обязательству.")
        else:
            if lev.get('type') == 'int':
                lines.append(f"* ⏱️ **[УТВЕРЖДЕНО] {lev['title']}:** {val} мес.")
            else:
                lines.append(f"* ⚙️ **[УТВЕРЖДЕНО] {lev['title']}:** {val}")
            lines.append(f"  * Зафиксировано стороной @{owner}.")

    pending_demands = [d for d in state.get("demands", []) if d.get("status") == "pending"]
    if pending_demands:
        lines.append("\n### 3. Несогласованные встречные требования сторон:")
        for d in pending_demands:
            if d.get("demand_type") == "pledge":
                lines.append(f"* ⚠️ **Требование @{d['from']} к @{d['target']}:** взнос {format_money(d['value'])} (не согласован).")
            else:
                v_str = "вкл" if d.get("value") is True else ("выкл" if d.get("value") is False else str(d.get("value")))
                lines.append(f"* ⚠️ **Требование @{d['from']} к @{d['target']}:** {d.get('title')} ({v_str}) (не утверждено).")

    return "\n".join(lines)


def render_voter_checklist(speaker: str, profile: dict, state: dict, levers_config: list[dict], total_cost: float) -> str:
    """Формирует персональный чек-лист соответствия условиям для голосующей стороны."""
    attrs = profile.get("attrs", {})
    p_state = state.get("players", {}).get(speaker, {})
    my_contrib = p_state.get("contribution") or 0.0
    my_share = p_state.get("share") or 0.0
    my_spent_levers = p_state.get("spent_on_levers", 0.0)
    base_budget = profile.get("money", 0)

    total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
    deficit = max(0.0, total_cost - total_c)

    lines = [
        f"🎯 ПЕРСОНАЛЬНЫЙ ЧЕК-ЛИСТ ДЛЯ СТОРОНЫ: @{speaker}",
        f"1. Твои затраты: взнос {format_money(my_contrib)} + спецобязательства {format_money(my_spent_levers)} = {format_money(my_contrib + my_spent_levers)} (из бюджета {format_money(base_budget)}).",
        f"2. Твоя доля в управлении: {my_share:g}% (минимум: {attrs.get('min_share', 0):g}%).",
        f"3. Статус бюджета проекта: {'⚠️ ДЕФИЦИТ ' + format_money(deficit) + ' (Принятие с дефицитом допустимо большинством голосов, покрытие ляжет на резервные фонды)' if deficit > 100_000.0 else '✅ ПОЛНОСТЬЮ ЗАКРЫТ (100% средств обеспечено)'}.",
        "4. Статус ключевых нефинансовых пунктов (обрати внимание на утвержденные гарантии в Разделе 2 выше!):",
    ]
    current_levers = state.get("levers", {})
    for lev in levers_config:
        val = current_levers.get(lev["id"], lev.get("default"))
        l_type = lev.get("type", "bool")
        if l_type == "bool":
            val_str = "УТВЕРЖДЕНО ✅" if val is True else "НЕ ПРИНЯТО ❌"
        elif l_type == "int":
            val_str = f"УТВЕРЖДЕНО ({val} мес.) ✅" if val else "НЕ ПРИНЯТО ❌"
        elif l_type == "str":
            val_str = f"УТВЕРЖДЕНО ({val}) ✅" if val and str(val).lower() != "none" else "НЕ ОПРЕДЕЛЕНО ❌"
        else:
            val_str = f"УТВЕРЖДЕНО ({val}) ✅" if val else "НЕ ПРИНЯТО ❌"
        lines.append(f"   - {lev['title']}: {val_str}")

    my_demands = [d for d in state.get("demands", []) if d.get("from") == speaker]
    if my_demands:
        lines.append("5. Статус твоих встречных требований к другим сторонам:")
        for d in my_demands:
            st_str = {"agreed": "СОГЛАСОВАНО ✅", "withdrawn": "ОТОЗВАНО"}.get(d.get("status"), "НЕ СОГЛАСОВАНО ⏳")
            if d.get("demand_type") == "pledge":
                lines.append(f"   - Взнос @{d['target']} ({format_money(d['value'])}): {st_str}")
            else:
                lines.append(f"   - {d.get('title')}: {st_str}")
        lines.append("   💡 Если второстепенные пожелания не удовлетворены, но твои главные красные линии соблюдены и бюджет закрыт — прояви конструктивную позицию и готовность к компромиссу (допустимо голосовать ЗА или ВОЗДЕРЖАЛСЯ).")

    lines.append(f"6. Твои обязательные красные линии: {', '.join(attrs.get('red_lines', [])) if attrs.get('red_lines') else 'Нет жестких ограничений (ориентируйся на общий баланс интересов)'}.")
    return "\n".join(lines)


def render_history(history: list) -> str:
    if not history:
        return "(начало переговоров)"
    blocks = []
    for item in history:
        action = item.get("action", {})
        act_type = action.get("type")
        speaker = item.get("speaker")
        target = item.get("target")
        target_str = f" ➔ @{target}" if target else ""
        t_num = item.get("turn")

        if act_type == "ACTION":
            app = action.get("applied", [])
            details = []
            for a in app:
                if a.get("kind") == "PLEDGE":
                    c_val = a.get("contribution")
                    s_val = a.get("share", 0)
                    details.append(f"Взнос: {format_money(c_val)} ({s_val:g}%)")
                elif a.get("kind") == "LEVER":
                    val_str = "ВКЛЮЧЕНО" if a.get("value") is True else ("ВЫКЛЮЧЕНО" if a.get("value") is False else str(a.get("value")))
                    details.append(f"{a.get('title')}: {val_str}")
                elif a.get("kind") == "DEMAND":
                    d_type = a.get("demand_type")
                    t_p = a.get("target", "всем")
                    if d_type == "pledge":
                        details.append(f"Требование к @{t_p}: взнос {format_money(a.get('value'))}")
                    else:
                        val_str = "ВКЛ" if a.get("value") is True else ("ВЫКЛ" if a.get("value") is False else str(a.get("value")))
                        details.append(f"Требование к @{t_p}: {a.get('title')} ({val_str})")
                elif a.get("kind") == "REJECTED":
                    details.append(f"⚠️ {a.get('reason')}")

            detail_str = f" [{'; '.join(details)}]" if details else ""
            msg = action.get("message", "").strip()
            if msg:
                blocks.append(f"[{t_num}] {speaker}{target_str}:{detail_str} «{msg}»")
            else:
                blocks.append(f"[{t_num}] {speaker}{target_str}:{detail_str}")
        elif act_type == "VOTE":
            blocks.append(f"[{t_num}] {speaker}: VOTE {action.get('vote')} — «{action.get('comment', '')}»")
        elif act_type == "SKIPPED":
            blocks.append(f"[{t_num}] {speaker}: ход пропущен (invalid_response)")
        else:
            blocks.append(f"[{t_num}] {speaker}: {item.get('command')}")
    return "\n".join(blocks)


def format_action_badge(actions: list) -> str:
    """Формирует строку со всеми зафиксированными действиями и требованиями."""
    items = []
    for a in actions:
        k = a.get("kind")
        if k == "PLEDGE":
            items.append(f"Взнос: {format_money(a.get('contribution'))}")
        elif k == "LEVER":
            val = a.get("value")
            val_str = "да" if val is True else ("нет" if val is False else str(val))
            items.append(f"{a.get('title')}: {val_str}")
        elif k == "DEMAND":
            d_type = a.get("demand_type")
            target = a.get("target", "всем")
            if is_owner(target, a.get("from", "")):
                continue
            if d_type == "pledge":
                items.append(f"Требование к @{target}: взнос {format_money(a.get('value'))}")
            else:
                v = a.get("value")
                v_str = "вкл" if v is True else ("выкл" if v is False else str(v))
                items.append(f"Требование к @{target}: {a.get('title')} ({v_str})")
    return f" 📝 *[{', '.join(items)}]*" if items else ""


def render_dialogue(history, state, voting_summary, total_cost, levers_config):
    total_money = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
    deficit = max(0.0, total_cost - total_money)

    lines = [
        "# 🏛️ ПРОТОКОЛ И СТЕНОГРАММА ОБЩЕСТВЕННЫХ СЛУШАНИЙ",
        "**Повестка:** Согласование комплексного плана реконструкции очистных сооружений и ликвидации экологического кризиса в городе N",
        f"**Состав участников:** {', '.join(state['players'].keys())}\n",
    ]

    phase1_items = [h for h in history if h.get("phase") == 1]
    phase2_items = [h for h in history if h.get("phase") == 2]
    phase3_items = [h for h in history if h.get("phase") == 3]

    if phase1_items:
        lines.append("## 📌 ЭТАП 1. Официальные вступительные заявления сторон\n")
        for h in phase1_items:
            t = h["turn"]
            sp = h["speaker"]
            act = h.get("action", {})
            if act.get("status") == "invalid_response" or act.get("type") == "SKIPPED":
                lines.append(f"**[{t}] {sp}:**\n\n> ⚠️ **Ход пропущен:** корректный ответ участника не получен.\n")
                continue
            msg = act.get("message", "")
            app = act.get("applied", [])
            badge = format_action_badge(app)
            lines.append(f"**[{t}] {sp}:**{badge}\n  «{msg}»\n")

    if phase2_items:
        lines.append("\n---\n## 💬 ЭТАП 2. Переговоры, поправки и распределение обязательств\n")
        has_announced_last_call = False
        for h in phase2_items:
            t = h["turn"]
            sp = h["speaker"]
            target = h.get("target")
            is_lc = h.get("is_last_call", False)

            if is_lc and not has_announced_last_call:
                lines.append("### 🔔 Финальный раунд согласования (Last Call)\n")
                has_announced_last_call = True

            if is_lc:
                sp_label = f"**[{t}] {sp} (Last Call):**"
            elif target:
                sp_label = f"**[{t}] {sp} ➔ @{target}:**"
            else:
                sp_label = f"**[{t}] {sp}:**"

            act = h.get("action", {})
            if act.get("status") == "invalid_response" or act.get("type") == "SKIPPED":
                lines.append(f"{sp_label}\n\n> ⚠️ **Ход пропущен:** корректный ответ участника не получен.\n")
                continue
            msg = act.get("message", "")
            app = act.get("applied", [])
            badge = format_action_badge(app)
            lines.append(f"{sp_label}{badge}\n  «{msg}»\n")

    treaty_doc = render_draft_treaty(state, levers_config, total_cost)

    if phase3_items:
        lines.append("\n---\n## 📋 Проект Соглашения сторон, вынесенный на голосование\n")
        lines.append(treaty_doc + "\n")
        lines.append("\n---\n## 🗳️ ЭТАП 3. Поименное голосование по проекту Соглашения\n")
        for h in phase3_items:
            t = h["turn"]
            sp = h["speaker"]
            act = h.get("action", {})
            if act.get("status") == "invalid_response" or not act.get("vote"):
                lines.append(f"* **[{t}] {sp}:** ⚠️ **ГОЛОС НЕ ПОЛУЧЕН.**\n")
                continue
            vote = act["vote"]
            comment = act.get("comment", "")
            icon = "🟢" if vote == "ЗА" else ("🔴" if vote == "ПРОТИВ" else "⚪")
            lines.append(f"* **[{t}] {sp}:** {icon} **VOTE {vote}:** «{comment}»\n")

    passed = voting_summary.get("passed", False)
    votes_for = voting_summary.get("votes_for", 0)
    votes_against = voting_summary.get("votes_against", 0)
    votes_abstain = voting_summary.get("votes_abstain", 0)
    quorum = voting_summary.get("quorum", 5)
    majority_threshold = voting_summary.get("majority_threshold", (quorum // 2) + 1)
    deficit = voting_summary.get("deficit", max(0.0, total_cost - total_money))

    lines.append("\n---\n## 📜 ИТОГОВЫЙ ПРОТОКОЛ ОБЩЕСТВЕННЫХ СЛУШАНИЙ ГОРОДА N\n")
    if passed:
        lines.append("🎉 **СТАТУС: ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ!**\n")
    else:
        lines.append("❌ **СТАТУС: ПРОТОКОЛ НЕ ПРИНЯТ!**\n")
    lines.append(
        f"* **Итоги голосования:** За: **{votes_for}**, Против: **{votes_against}**, Воздержались: **{votes_abstain}** (Порог большинства: {majority_threshold} из {quorum})\n"
        f"* **Финансирование:** {format_money(total_money)} из {format_money(total_cost)} ({((total_money/total_cost)*100) if total_cost else 0:.1f}% от целевой суммы)\n"
    )
    if passed:
        if deficit > 100_000.0:
            lines.append(
                f"> ⚠️ **Примечание по финансированию:** Проект принят при наличии остаточного дефицита в размере {format_money(deficit)}. "
                f"Недостающие средства подлежат согласованию отдельным протоколом из резервных фондов или дополнительных источников.\n"
            )
    else:
        reasons = []
        if votes_for < majority_threshold:
            reasons.append(f"не набрано большинство голосов (За: {votes_for}, необходимо: {majority_threshold})")
        if deficit > 100_000.0:
            reasons.append(f"не закрыт дефицит бюджета проекта ({format_money(deficit)})")
        reason_str = "; ".join(reasons) if reasons else "несогласованность ключевых условий"
        lines.append(f"* **Причина отклонения:** {reason_str}.\n")

    lines.append("### 1. Согласованный финансовый план:")
    lines.append("| Участник | Согласованный взнос | Доля в проекте |")
    lines.append("| :--- | :--- | :--- |")

    total_share = 0
    for name, st in state["players"].items():
        c = st.get("contribution")
        s = st.get("share")
        c_val = c if c is not None else 0.0
        c_str = format_money(c_val)
        s_str = f"{s:g}%" if s is not None and s > 0 else "0%"
        if s:
            total_share += s
        lines.append(f"| **{name}** | {c_str} | {s_str} |")

    lines.append(f"\n* **Всего утверждено финансирования:** {format_money(total_money)} (из требуемых {format_money(total_cost)})")
    lines.append(f"* **Суммарная распределенная доля:** {total_share:g}%\n")

    lines.append("### 2. Утвержденные условия и обязательства сторон:")
    current_levers = state.get("levers", {})
    agreed_count = 0
    for lev in levers_config:
        lid = lev["id"]
        val = current_levers.get(lid, lev.get("default"))
        l_type = lev.get("type", "bool")
        title = lev.get("title", lid)
        owner = lev.get("owner", "")
        desc = lev.get("description", "")
        desc_str = f" ({desc})." if desc else "."

        icon = "⚙️"
        t_lower = title.lower()
        if "вины" in t_lower or "ответственност" in t_lower:
            icon = "⚖️"
        elif "мощност" in t_lower or "пром" in t_lower:
            icon = "🏭"
        elif "срок" in t_lower or "график" in t_lower:
            icon = "⏱️"
        elif "тариф" in t_lower or "льгот" in t_lower:
            icon = "💰"
        elif "датчик" in t_lower or "мониторинг" in t_lower:
            icon = "📡"

        if l_type == "bool":
            if val:
                lines.append(f"* ✅ **{title}:** Подтверждено стороной @{owner}{desc_str}")
            else:
                lines.append(f"* ❌ **{title}:** Не согласовано сторонами.")
        else:
            agreed_count += 1
            unit = " мес." if l_type == "int" else ""
            lines.append(f"* {icon} **{title}:** Утверждено на уровне: **{val}{unit}** (зафиксировано стороной @{owner}).")

    return "\n".join(lines)


# --- negotiation_game/run_store.py ---

"""Локальное сохранение прогресса игры после каждого хода и при прерывании."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunStore:
    def __init__(self):
        self.run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex[:8]
        root = Path(os.environ.get("NEGOTIATION_GAMES_DIR", "games")).expanduser().resolve()
        self.directory = root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = {"result.json": str(self.directory / "result.json"),
                      "result.md": str(self.directory / "result.md")}

    def save(self, payload: dict) -> None:
        payload["run_id"] = self.run_id
        payload["files"] = dict(self.files)
        contents = {"result.json": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    "result.md": payload["report"]}
        for name, content in contents.items():
            target = self.directory / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(content + "\n", encoding="utf-8")
            temporary.replace(target)


# --- negotiation_game/langflow_component.py ---

"""Адаптер Langflow и оркестрация этапов основной игры переговоров."""

import base64
import asyncio
import copy
import json
import random
import zlib
from typing import Any

from pydantic import BaseModel, ValidationError
from langchain_core.exceptions import OutputParserException

from lfx.custom import Component
from lfx.io import DataInput, IntInput, MultilineInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message
from langchain_core.messages import HumanMessage, SystemMessage


DEFAULT_PROJECT_COST = 270_000_000.0  # 270 млн руб. по умолчанию

RAG_ROLE_BY_PLAYER_ID = {
    "city_administration": "admin",
    "industrial_holding": "holding",
    "regional_government": "region",
    "city_residents": "residents",
    "environmental_ngo": "ngo",
}


def _rag_role(profile: dict, speaker: str) -> str:
    player_id = profile.get("id")
    if not isinstance(player_id, str) or player_id not in RAG_ROLE_BY_PLAYER_ID:
        raise ValueError(f"Неизвестная RAG-роль: игрок {speaker!r}, id={player_id!r}")
    return RAG_ROLE_BY_PLAYER_ID[player_id]


def _rag_chunk_text(record: dict) -> str:
    """Текст фактов без каталогового заголовка и технической обёртки.

    Убираются только точные заголовки/примечание из метаданных записи.
    Сами факты, ограничения роли и ссылки внутри документа сохраняются.
    """
    text = record["text"].strip()
    title = record.get("document_title")
    section = record.get("section_title")
    if title:
        text = text.removeprefix(title + "\n")
    if section:
        text = text.removeprefix(f"{record['chunk_id']}. {section}\n")
    note = record.get("authority_note")
    if note:
        text = text.removesuffix("\nКонтекст использования: " + note)
    return text.strip()


def search_enabled(record: dict) -> bool:
    """Allow explicit main-index entries and legacy canonical material only."""
    if "index_policy" in record:
        return record["index_policy"] == "main_index"
    return record.get("area") in ("shared", "role", "judge", "methodology")


def _get_default_settings() -> dict:
    """Дефолтные параметры сценария (если узел GameSettings не подключен)."""
    return {
        "total_cost": DEFAULT_PROJECT_COST,
        "levers": [
            {
                "id": "guilt",
                "title": "Признание вины за аварию",
                "aliases": ["вина", "виновность", "ответственность_за_аварию", "guilt"],
                "type": "str",
                "allowed": ["shared", "holding", "city"],
                "owner": "Региональная власть",
                "cost": 0.0,
                "default": "shared",
                "description": "Юридическое признание вины: shared (обоюдная вина — компромисс), holding (вина завода), city (вина города)",
            },
            {
                "id": "plant_mode",
                "title": "Режим работы предприятия",
                "aliases": ["режим", "мощность", "режим_завода", "plant_mode"],
                "type": "str",
                "allowed": ["100%", "70%"],
                "owner": "Промышленный холдинг",
                "cost": 0.0,
                "default": "100%",
                "description": "Режим мощности комбината: 100% (сохранение занятости и выручки), 70% (временное ограничение для снижения сбросов)",
            },
            {
                "id": "timeline",
                "title": "Срок реконструкции (мес.)",
                "aliases": ["срок", "время", "месяцы", "timeline", "timeline_months"],
                "type": "int",
                "owner": "Промышленный холдинг",
                "cost": 0.0,
                "default": 36,
                "min_value": 24,
                "max_value": 48,
                "description": "Срок ввода пусковых комплексов очистных сооружений (24, 36 или 48 месяцев)",
            },
            {
                "id": "tariffs_freeze",
                "title": "Заморозка тарифов ЖКХ",
                "aliases": ["тарифы", "заморозка", "заморозка_тарифов", "tariffs", "tariffs_freeze"],
                "type": "bool",
                "owner": "Администрация города N",
                "cost": 0.0,
                "default": False,
                "description": "Мораторий на повышение тарифов на водоснабжение и водоотведение для населения города",
            },
            {
                "id": "tax_relief",
                "title": "Налоговые льготы предприятию",
                "aliases": ["льготы", "налоги", "налоговые_льготы", "tax", "tax_relief"],
                "type": "bool",
                "owner": "Региональная власть",
                "cost": 0.0,
                "default": False,
                "description": "Региональные налоговые преференции и льгота по налогу на имущество для холдинга на период инвестпроекта",
            },
            {
                "id": "open_sensors",
                "title": "Открытые эко-датчики 24/7",
                "aliases": ["датчики", "мониторинг", "экомониторинг", "сенсоры", "open_sensors", "sensors"],
                "type": "bool",
                "owner": "Экологическая организация (НКО)",
                "cost": 0.0,
                "default": False,
                "description": "Установка автоматических станций онлайн-контроля сбросов в реку с открытой публикацией данных 24/7",
            },
            {
                "id": "protest_moratorium",
                "title": "Мораторий на протесты и судебные иски",
                "aliases": ["протесты", "мораторий", "митинги", "иски", "protest_moratorium", "moratorium"],
                "type": "bool",
                "owner": "Инициативная группа жителей",
                "cost": 0.0,
                "default": False,
                "description": "Отказ жителей и активистов от перекрытия дорог, несанкционированных акций и отзыв коллективных исков в суд",
            },
            {
                "id": "water_supply",
                "title": "Гарантия питьевой воды жителям",
                "aliases": ["вода", "водоснабжение", "питьевая_вода", "water", "water_supply"],
                "type": "bool",
                "owner": "Администрация города N",
                "cost": 20_000_000.0,
                "default": False,
                "description": "Внебюджетная доставка чистой питьевой воды жителям моногорода до ввода сооружений",
            },
        ],
    }


class NegotiationGame(Component):
    display_name = "Negotiation Game"
    description = "Оркестратор общественных слушаний: 3-фазный регламент (Вводные заявления -> Торг -> Поименное голосование большинством)."
    icon = "messages-square"

    inputs = [
        DataInput(name="rag_config", display_name="Общий RAG", required=True),
        DataInput(
            name="players",
            display_name="Players",
            required=True,
            is_list=True,
        ),
        IntInput(
            name="turns",
            display_name="Максимум ходов торга (Фаза 2)",
            value=15,
            required=True,
        ),
        IntInput(name="request_timeout_seconds", display_name="Лимит запроса, секунд",
                 value=60, required=True, advanced=True),
        IntInput(name="game_timeout_seconds", display_name="Лимит игры, секунд",
                 value=900, required=True, advanced=True),
        IntInput(name="request_attempts", display_name="Попыток при сетевой ошибке",
                 value=2, required=True, advanced=True),
        MultilineInput(
            name="rules_prompt",
            display_name="Инструкция регламента слушаний",
            info="Промпт правил и регламента 3 фаз общественных слушаний.",
            value="",
            required=False,
        ),
        DataInput(
            name="settings",
            display_name="Настройки игры",
            info="Параметры сценария, бюджет и переговорные условия (опционально)",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Game Result",
            method="run_chat",
            types=["Data"],
        )
    ]


    def _player_turn_arguments(self, result: dict) -> dict:
        """Читает аргументы единственного tool call до Pydantic-валидации.

        Не извлекает действия из речи и не восстанавливает повреждённый JSON.
        """
        if not isinstance(result, dict):
            raise ValueError("Ожидался результат tool call PlayerTurn")
        raw = result.get("raw")
        tool_calls = raw.get("tool_calls", []) if isinstance(raw, dict) else getattr(raw, "tool_calls", [])
        invalid = raw.get("invalid_tool_calls", []) if isinstance(raw, dict) else getattr(raw, "invalid_tool_calls", [])
        if tool_calls or invalid:
            calls = list(tool_calls or []) + list(invalid or [])
            if len(calls) != 1 or calls[0].get("name") != "PlayerTurn":
                raise ValueError("Верни ровно один вызов функции PlayerTurn")
            arguments = calls[0].get("args")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Аргументы PlayerTurn должны быть JSON-объектом")
            return arguments
        parsed = result.get("parsed")
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Модель не вернула аргументы функции PlayerTurn: "
                         + str(result.get("parsing_error") or "tool call отсутствует"))

    async def _structured_turn(self, current: dict, phase: str, turn: int, messages: list,
                         state: dict, levers_config: list[dict], total_cost: float,
                         voting: bool = False) -> tuple[PlayerTurn | None, dict]:
        """Один ответ и одна попытка исправления; без текстового парсера."""
        # method=function_calling не требует нативной строгой JSON Schema у провайдера.
        # Если модель не поддерживает tools, завершаем запуск с явной ошибкой.
        llm = current["llm"]
        timeout = float(getattr(self, "request_timeout_seconds", 60))
        # ChatOpenAI уже содержит созданные SDK-клиенты. Одного model_copy с
        # max_retries=0 недостаточно: копируем настройки самих клиентов тоже.
        if isinstance(llm, BaseModel) and getattr(llm, "root_async_client", None) is not None:
            root = llm.root_client.with_options(timeout=timeout, max_retries=0)
            async_root = llm.root_async_client.with_options(timeout=timeout, max_retries=0)
            llm = llm.model_copy(update={"request_timeout": timeout, "max_retries": 0,
                "root_client": root, "client": root.chat.completions,
                "root_async_client": async_root, "async_client": async_root.chat.completions})
        structured = llm.with_structured_output(
            player_turn_tool_schema(state["player_ids"], voting, levers_config),
            method="function_calling", include_raw=True)
        # Имена владельцев в настройках могут быть полными названиями ролей.
        # Разрешаем их тем же правилом, что использует игровой валидатор.
        contract_levers = [{**lever, "owner": next(
            (name for name in state["player_ids"].values() if is_owner(lever.get("owner", ""), name)),
            lever.get("owner"))} for lever in levers_config]
        contract = (
            "Верни один вызов функции PlayerTurn строго по её схеме. speech — публичная речь без команд. "
            "Все ссылки на участников — точные player_id из списка, не имена. "
            "Деньги во входном состоянии — рубли, в ответе — миллионы. "
            "Формат вызова отличается от внутреннего STATE: null в аргументах не нужен. "
            "Собственный взнос: own_contribution={action:unchanged, amount_millions:0} — без изменений; "
            "action=set задаёт новый взнос, включая 0 для обнуления. "
            "Собственные условия меняются через own_conditions: список lever_id и value_json. "
            "Условия своей роли помещай в own_conditions, не требуй их от себя или постороннего участника. "
            "Денежные требования — только pledge_demands с amount_millions. "
            "Требования условий — только condition_demands с lever_id и value_json. "
            "value_json — значение условия строкой: true/false для bool, 24 для int, "
            "70% или holding для строкового условия. Дополнительные кавычки не нужны. "
            "condition_demands.target — player_id другого владельца этого условия из списка ниже. "
            "Пустые списки — без действий. "
            "Требования имеют ID от Python. Отвечать может только адресат, отзывать — только автор. "
            "accept не меняет обязательство автоматически: явно заполни собственный взнос или условие. "
            "Для нового требования counter_to — пустая строка; для встречного — точный ID исходного. "
            "Вне голосования vote=none. На голосовании только речь и vote=for/against/abstain; "
            "own_contribution.action=unchanged, amount_millions=0; все списки пустые. "
            "Обращение ко всем — addressed_to=everyone, конкретному участнику — его player_id. "
            "Не извлекай команды из речи: действия задаются полями.\n"
            "УЧАСТНИКИ (player_id → имя): " + json.dumps(state["player_ids"], ensure_ascii=False)
            + "\nТРЕБОВАНИЯ (суммы в рублях): " + json.dumps(state["demands"], ensure_ascii=False)
            + "\nУСЛОВИЯ (ID, тип и владелец): " + json.dumps(
                [{k: lever.get(k) for k in ("id", "type", "owner", "allowed", "min_value", "max_value")}
                 for lever in contract_levers], ensure_ascii=False)
            + "\nТЕКУЩИЙ ЭТАП: " + ("голосование" if voting else "переговоры, vote=none")
            + "\n" + player_turn_examples(state["player_ids"], contract_levers, voting, current["name"])
        )
        rules = str(getattr(self, "rules_prompt", "") or "").strip()
        role = (
            f"Ты представляешь сторону «{current['name']}».\n"
            f"РОЛЬ И СТРАТЕГИЯ:\n{current['prompt']}\n\n"
            "В speech пиши только публичную реплику участника слушаний. "
            "Используй сведения о сценарии и ограничения своей роли для принятия решений. "
            "Служебные инструкции, технические поля и документы целиком в речь не переноси. "
            "Числа и лимиты бери из текущего состояния игры; справочные примеры не создают обязательств.\n"
        )
        if rules:
            role += "\nРЕГЛАМЕНТ ЗАСЕДАНИЯ:\n" + rules + "\n"
        request = [SystemMessage(content=role + "\n" + contract), *messages]
        wrapped = {**current, "llm": structured}
        for attempt in range(2):
            result = await self._invoke_model(wrapped, phase, turn, request)
            arguments = None
            try:
                arguments = self._player_turn_arguments(result)
                self._debug_calls[-1].update(raw_arguments=copy.deepcopy(arguments), normalization=[])
                normalized, changes = normalize_player_turn(
                    arguments, state["player_ids"], self._debug_calls[-1]["normalization"], levers_config)
                self._debug_calls[-1].update(normalized_arguments=copy.deepcopy(normalized), normalization=changes)
                answer = player_turn_schema().model_validate(normalized)
                candidate, action = prepare_player_turn(
                    answer, current["name"], state, levers_config, total_cost, voting)
            except (ValidationError, ValueError, OutputParserException) as exc:
                self._debug_calls[-1]["validation_error"] = str(exc)
                errors = (exc.errors(include_url=False, include_context=False, include_input=False)
                          if isinstance(exc, ValidationError) else [{"loc": [], "msg": str(exc)}])
                self._debug_calls[-1]["validation_errors"] = errors
                if attempt == 0:
                    request.append(HumanMessage(content=(
                        "Ответ не применён. Исправь весь PlayerTurn и верни один вызов функции.\n"
                        "Исходные аргументы JSON: " + json.dumps(arguments, ensure_ascii=False)
                        + "\nОшибки по полям: " + json.dumps(errors, ensure_ascii=False)
                        + "\nДопустимые player_id: " + json.dumps(list(state["player_ids"]), ensure_ascii=False)
                        + "\nДля обращения ко всем используй addressed_to=everyone. Не угадывай ID. "
                        "Не меняй смысл своей позиции ради валидации. Денежные требования запиши "
                        "в pledge_demands, требования условий — в condition_demands. "
                        "value_json — значение строкой: true, false, 36, 70%, holding. "
                        "Только точные значения из ограничений условия. "
                        "Собственные условия запиши в own_conditions, требования — другому владельцу условия. "
                        "counter_to для нового требования — пустая строка. "
                        + ("На голосовании меняются только speech и vote.\n" if voting else "Вне голосования vote=none.\n")
                        + player_turn_examples(state["player_ids"], contract_levers, voting, current["name"]))))
                    continue
                return None, {"type": "VOTE" if voting else "SKIPPED",
                              "vote": None, "status": "invalid_response", "applied": [],
                              "message": "Ход пропущен: корректный ответ не получен.",
                              "comment": "Корректный голос не получен.", "target": None}
            state.clear()
            state.update(candidate)
            return answer, action

    async def _invoke_model(self, current: dict, phase: str, turn: int, messages: list):
        call = {
            "phase": phase,
            "turn": turn,
            "player": current["name"],
            "messages": [{"role": message.type, "content": str(message.content)} for message in messages],
        }
        self._debug_calls.append(call)

        max_attempts = int(getattr(self, "request_attempts", 2))
        timeout = float(getattr(self, "request_timeout_seconds", 60))
        for attempt in range(1, max_attempts + 1):
            call["attempt"] = attempt
            self.status = f"{phase}: {current['name']}, ход {turn}, попытка {attempt}"
            self._save_progress()
            try:
                async with asyncio.timeout(timeout):
                    response = await current["llm"].ainvoke(messages)
                raw = response.get("raw") if isinstance(response, dict) else response
                content = raw.model_dump(mode="json") if isinstance(raw, BaseModel) else getattr(raw, "content", raw)
                call["response"] = content if isinstance(content, str) else str(content)
                # Пауза между запросами для предотвращения превышения burst-лимитов API (GigaChat / OpenAI)
                await asyncio.sleep(1.0)
                return response
            except asyncio.CancelledError:
                call["status"] = "cancelled"
                raise
            except Exception as exc:
                err_str = str(exc).lower()
                err_name = type(exc).__name__

                # Определяем ошибки rate limit (429) или временные сбои шлюза / сети
                is_rate_limit = (
                    "429" in err_str
                    or "rate limit" in err_str
                    or "ratelimit" in err_name.lower()
                    or "too many requests" in err_str
                )
                status_code = getattr(exc, "status_code", None)
                is_transient = isinstance(exc, (TimeoutError, ConnectionError)) or is_rate_limit or status_code in {500, 502, 503, 504} or any(
                    sig in err_str
                    for sig in [
                        "500", "502", "503", "504",
                        "timeout", "timed out", "connection",
                        "remote disconnected", "server disconnected", "overloaded"
                    ]
                )

                call["error"] = f"{err_name}: {exc}"
                if attempt < max_attempts and is_transient:
                    # Извлекаем retry-after если сервер вернул конкретное время ожидания
                    retry_after = None
                    resp_obj = getattr(exc, "response", None)
                    if resp_obj is not None:
                        hdrs = getattr(resp_obj, "headers", None)
                        if hdrs:
                            ra_hdr = hdrs.get("retry-after") or hdrs.get("Retry-After")
                            if ra_hdr:
                                try:
                                    retry_after = float(ra_hdr)
                                except (ValueError, TypeError):
                                    pass

                    if retry_after is not None and retry_after > 0:
                        wait_time = min(retry_after, 10.0)
                    else:
                        wait_time = min(2.0 ** attempt + random.uniform(0, 0.5), 10.0)

                    call.setdefault("retries", []).append({
                        "attempt": attempt,
                        "error": f"{err_name}: {exc}",
                        "wait_time": round(wait_time, 2),
                    })
                    self._save_progress()
                    await asyncio.sleep(wait_time)
                else:
                    if is_rate_limit:
                        raise RuntimeError(
                            f"Превышен лимит запросов к модели ({err_name}: 429 Rate Limit) для игрока '{current['name']}'. "
                            f"Было выполнено {attempt} попыток с паузами. Проверьте ограничения перед повторным запуском "
                            f"или проверьте квоту/баланс токенов в личном кабинете GigaChat."
                        ) from exc
                    if isinstance(exc, TimeoutError):
                        raise TimeoutError(
                            f"Модель {current['name']} не ответила за {timeout:g} секунд"
                        ) from exc
                    raise

    def _save_progress(self, status="running", error=None):
        store = getattr(self, "_run_store", None)
        if store is None:
            return None
        state = self._live_state
        total_cost = self._live_cost
        history = self._live_history
        payload = {
            "status": status, "error": error, "state": copy.deepcopy(state),
            "history": copy.deepcopy(history), "total_cost": total_cost,
            "total_money": sum(p.get("contribution") or 0 for p in state["players"].values()),
            "consensus_reached": None, "voting_summary": None,
            "progress": str(getattr(self, "status", "")),
            "players": [{k: v for k, v in p["profile"].items() if k != "llm"}
                        for p in self._live_players],
            "rag_retrievals": copy.deepcopy(self._rag_audit),
            "debug": self._debug_data(history, state),
            "report": render_partial_report(history, state, total_cost,
                                             self._live_levers, status, error),
        }
        store.save(payload)
        return Data(data=payload, text=payload["report"])

    def _debug_data(self, history: list, state: dict | None) -> dict:
        return {
            "calls": self._debug_calls,
            "initial_state": self._debug_initial_state,
            "updates": [
                {"turn": item["turn"], "phase": item["phase"], "player": item["speaker"],
                 "applied": item["action"].get("applied", []), "state": item["state"]}
                for item in history
            ],
            "last_state": copy.deepcopy(state),
        }


    # RAG, Контекст

    async def _rag_context(self, speaker, profile, history, phase, retrieve=True) -> str:
        if not getattr(self, "rag_config", None) or not getattr(self.rag_config, "data", None):
            return ""
        if not getattr(self, "use_rag", True):
            return ""

        role = _rag_role(profile, speaker)
        config = self.rag_config.data
        payload = config.get("corpus_b64")
        if not payload:
            raise ValueError("RAG: отсутствует корпус обязательных фактов и инструкций (corpus_b64)")
        if getattr(self, "_rag_corpus_payload", None) != payload:
            try:
                raw = zlib.decompress(base64.b64decode(payload)).decode("utf-8")
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                if any(not isinstance(record, dict)
                       or not isinstance(record.get("chunk_id"), str)
                       or not isinstance(record.get("text"), str)
                       or not isinstance(record.get("audience"), list) for record in records):
                    raise ValueError("Неверная структура фрагментов корпуса")
            except (ValueError, zlib.error, UnicodeError) as exc:
                raise ValueError("RAG: не удалось прочитать корпус обязательного контекста") from exc
            self._rag_corpus_records = records
            self._rag_corpus_payload = payload
        records = self._rag_corpus_records
        core = [record for record in records
                if record.get("area") in {"shared", "role"} and role in record.get("audience", [])]
        if not any(record.get("area") == "shared" for record in core) or not any(
                record.get("area") == "role" for record in core):
            raise ValueError(f"RAG: неполный обязательный контекст для роли {role!r}")
        core_ids = {record["chunk_id"] for record in core}
        allowed_extra = {record["chunk_id"]: record for record in records
                         if record.get("area") not in {"shared", "role"}
                         and role in record.get("audience", []) and search_enabled(record)}
        parts = [_rag_chunk_text(record) for record in core]

        if not hasattr(self, "_rag_audit"):
            self._rag_audit = []
        audit = {"speaker": speaker, "phase": phase, "role": role,
                 "core_chunk_ids": [record["chunk_id"] for record in core], "chunk_ids": [],
                 "retrieval_status": "skipped"}
        self._rag_audit.append(audit)
        if not retrieve:
            return "\n\n".join(parts)
        if not allowed_extra:
            audit["skip_reason"] = "no_allowed_sources"
            return "\n\n".join(parts)

        recent = "\n".join(str(h.get("action", {}).get("message", ""))[:900] for h in history[-2:])
        query = (f"{speaker}. Этап {phase}. {profile.get('intent', '')}\n"
                 f"Текущая тема переговоров: {recent}")[:6000]
        try:
            import chromadb
            from chromadb.config import Settings
            from openai import AsyncOpenAI
            key = config.get("embedding_api_key", "")
            if hasattr(key, "get_secret_value"):
                key = key.get_secret_value()
            async with AsyncOpenAI(base_url=config["embedding_base_url"], api_key=key, timeout=30, max_retries=0) as client:
                async with asyncio.timeout(30):
                    vector = (await client.embeddings.create(model=config["embedding_model"], input=query)).data[0].embedding
            if len(vector) != config["dimension"]:
                raise ValueError("Размерность запроса не совпадает с базой")
            db = chromadb.PersistentClient(path=config["chroma_path"], settings=Settings(anonymized_telemetry=False))
            collection = db.get_collection(config["collection_name"], embedding_function=None)
            result = collection.query(
                query_embeddings=[vector],
                n_results=config["top_k"],
                where={"$and": [{"visible_" + role: True}, {"area": {"$nin": ["shared", "role"]}},
                                 {"search_enabled": True}]},
                include=["documents", "metadatas", "distances"],
            )
            chunks = [allowed_extra[cid] for cid, meta in
                      zip(result["ids"][0], result["metadatas"][0])
                      if cid in allowed_extra and cid not in core_ids and search_enabled(meta)
                      and meta.get("area") not in {"shared", "role"}]
        except Exception as exc:
            audit["retrieval_status"] = "error"
            audit["error_type"] = type(exc).__name__
            return "\n\n".join(parts)

        audit["retrieval_status"] = "ok"
        audit["chunk_ids"] = [record["chunk_id"] for record in chunks]
        parts.extend(_rag_chunk_text(record)[:3500] for record in chunks)
        return "\n\n".join(parts)


    # ШАБЛОНЫ ПРОМПТОВ

    @staticmethod
    def _build_prompt_phase1(speaker, player_prompt, profile, state, history, rules_prompt, turn, total_phase1, levers_config, total_cost):
        return (
            f"=== 🏛️ ЭТАП 1 ИЗ 3: ВСТУПИТЕЛЬНЫЕ ЗАЯВЛЕНИЯ (Выступление {turn} из {total_phase1}) ===\n\n"
            f"Сторона: {speaker}\n"
            f"Участники: {', '.join(state['players'].keys())}\n\n"
            f"ЭКОНОМИЧЕСКИЙ ПРОФИЛЬ:\n{render_profile(profile, state, speaker)}\n\n"
            f"{render_player_levers(speaker, levers_config, state.get('levers', {}))}\n\n"
            f"{render_state(state, levers_config, total_cost)}"
        )

    @staticmethod
    def _build_prompt_phase2(speaker, player_prompt, profile, state, history, rules_prompt, turn_in_phase, max_phase_turns, global_turn, levers_config, total_cost):
        total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
        deficit = max(0.0, total_cost - total_c)
        banner = f"ЭТАП 2 ИЗ 3: ПЕРЕГОВОРЫ И ТОРГ (Раунд {turn_in_phase} из {max_phase_turns})"
        banner += f" | ДЕФИЦИТ: {format_money(deficit)}" if deficit > 1000 else " | БЮДЖЕТ СОБРАН"
        incoming = []
        for demand in state.get("demands", []):
            if demand.get("target") != speaker or demand.get("status") != "pending":
                continue
            value = format_money(demand["value"]) if demand.get("demand_type") == "pledge" else str(demand.get("value"))
            incoming.append(f"- @{demand['from']}: {demand.get('title', 'Взнос')} — {value}")
        demands_block = "\nТРЕБОВАНИЯ К ВАШЕЙ СТОРОНЕ:\n" + "\n".join(incoming) + "\n" if incoming else ""
        return (
            f"=== 💬 {banner} ===\n\n"
            f"Сторона: {speaker}\n"
            f"Участники: {', '.join(state['players'].keys())}\n\n"
            f"ЭКОНОМИЧЕСКИЙ ПРОФИЛЬ:\n{render_profile(profile, state, speaker)}\n\n"
            f"{render_player_levers(speaker, levers_config, state.get('levers', {}))}\n\n"
            f"{render_state(state, levers_config, total_cost)}\n"
            f"{demands_block}\n"
            f"ИСТОРИЯ ПЕРЕГОВОРОВ:\n{render_history(history)}"
        )

    @staticmethod
    def _build_prompt_last_call(speaker, player_prompt, profile, state, history, rules_prompt, turn_counter, levers_config, total_cost, deficit, available_money):
        return (
            f"=== 🔔 ФИНАЛЬНЫЙ РАУНД СОГЛАСОВАНИЯ / LAST CALL (Ход {turn_counter}) ===\n\n"
            f"Сторона: {speaker}\n"
            f"Участники: {', '.join(state['players'].keys())}\n\n"
            f"ЭКОНОМИЧЕСКИЙ ПРОФИЛЬ:\n{render_profile(profile, state, speaker)}\n\n"
            f"{render_player_levers(speaker, levers_config, state.get('levers', {}))}\n\n"
            f"{render_calculator(state, total_cost)}\n\n"
            f"Остаточный дефицит: {format_money(deficit)}.\n"
            f"Доступный остаток вашей стороны: {format_money(available_money)}."
        )

    @staticmethod
    def _build_prompt_phase3(speaker, player_prompt, profile, state, turn_in_phase, total_voters, majority_threshold, levers_config, total_cost):
        return (
            f"=== 🗳️ ЭТАП 3 ИЗ 3: ГОЛОСОВАНИЕ (Голос {turn_in_phase} из {total_voters}) ===\n\n"
            f"Сторона: {speaker}\n"
            f"Порог большинства: {majority_threshold} из {total_voters}.\n\n"
            f"{render_draft_treaty(state, levers_config, total_cost)}\n\n"
            f"{render_voter_checklist(speaker, profile, state, levers_config, total_cost)}"
        )
    def _init_session(self) -> tuple[list[dict], list[str], dict, dict]:
        player_data = self.players
        if not player_data:
            raise ValueError("Требуется хотя бы один игрок.")

        players = []
        for data in player_data:
            profile = data.data if hasattr(data, "data") else data
            name = str(profile.get("player_name") or profile.get("name") or "").strip()
            money_val = profile.get("budget_millions")
            if money_val is not None:
                try:
                    profile["money"] = float(money_val) * 1_000_000.0
                except (ValueError, TypeError):
                    pass
            elif "budget" in profile and "money" not in profile:
                profile["money"] = float(profile["budget"])
            prompt = str(profile.get("prompt", "")).strip()
            llm = profile.get("llm")

            if not name:
                raise ValueError("Имя игрока не может быть пустым.")
            if not prompt:
                raise ValueError(f"Промпт пуст для игрока '{name}'.")
            if llm is None:
                raise ValueError(f"Языковая модель не инициализирована для игрока '{name}'.")

            if (getattr(self, "use_rag", True)
                    and getattr(getattr(self, "rag_config", None), "data", None)):
                _rag_role(profile, name)

            players.append({
                "name": name,
                "llm": llm,
                "prompt": prompt,
                "profile": profile,
            })

        names_by_id = {}
        for player in players:
            player_id = player["profile"].get("id")
            if not isinstance(player_id, str) or not player_id or player_id in names_by_id:
                raise ValueError("У каждого игрока должен быть уникальный непустой id")
            names_by_id[player_id] = player["name"]
        names = [p["name"] for p in players]
        if len(names) != len(set(names)):
            raise ValueError("Имена игроков должны быть уникальными.")

        # Получение настроек сценария
        settings_data = None
        if getattr(self, "settings", None) and hasattr(self.settings, "data") and isinstance(self.settings.data, dict):
            settings_data = copy.deepcopy(self.settings.data)
        if not settings_data:
            settings_data = _get_default_settings()

        if "components" in settings_data and not settings_data.get("total_cost"):
            calc_cost = sum(c.get("cost", 0) for c in settings_data["components"])
            settings_data["total_cost"] = float(calc_cost) if calc_cost > 0 else DEFAULT_PROJECT_COST
        elif "total_cost" not in settings_data:
            settings_data["total_cost"] = DEFAULT_PROJECT_COST

        if "levers" not in settings_data or not settings_data["levers"]:
            settings_data["levers"] = _get_default_settings()["levers"]

        total_cost = float(settings_data["total_cost"])
        levers_config = settings_data["levers"]

        state = {
            "player_ids": names_by_id,
            "next_demand_id": 1,
            "total_cost": total_cost,
            "players": {
                p["name"]: {
                    "contribution": None,
                    "share": None,
                    "spent_on_levers": 0.0,
                    "max_budget": float(p["profile"].get("money", 0.0)),
                }
                for p in players
            },
            "levers": {
                lev["id"]: lev.get("default")
                for lev in levers_config
            },
            "demands": [],
        }

        return players, names, state, settings_data

    async def _run_phase1(self, players: list[dict], state: dict, history: list[dict], rules: str, turn_counter: int, levers_config: list[dict], total_cost: float) -> int:
        num_players = len(players)
        for p_idx in range(num_players):
            turn_counter += 1
            current = players[p_idx]
            speaker = current["name"]

            prompt = self._build_prompt_phase1(
                speaker=speaker,
                player_prompt=current["prompt"],
                profile=current["profile"],
                state=state,
                history=history,
                rules_prompt=rules,
                turn=turn_counter,
                total_phase1=num_players,
                levers_config=levers_config,
                total_cost=total_cost,
            )

            system_message = (
                "Ты участник общественных слушаний по экологическому кризису в городе N. "
                "ИДЕТ ЭТАП 1: ВСТУПИТЕЛЬНЫЕ ЗАЯВЛЕНИЯ СТОРОН. "
                "Официально озвучь стартовую позицию своей роли, финансовые рамки и ключевые требования. "
                "Заяви свой планируемый взнос и начальные параметры. "
                "Обязательства и требования фиксируются из структурированных полей ответа."
            )

            prompt += "\n\n" + await self._rag_context(speaker, current["profile"], history, 1)

            answer, action = await self._structured_turn(
                current, "Вступление", turn_counter,
                [SystemMessage(content=system_message), HumanMessage(content=prompt)],
                state, levers_config, total_cost,
            )
            command = answer.speech if answer is not None else ""
            history.append({
                "turn": turn_counter,
                "phase": 1,
                "speaker": speaker,
                "target": action.get("target"),
                "command": command,
                "structured_response": answer.model_dump() if answer is not None else None,
                "action": action,
                "state": copy.deepcopy(state),
            })
            self._save_progress()
        return turn_counter

    async def _run_phase2(self, players: list[dict], state: dict, history: list[dict], rules: str, turn_counter: int, max_turns: int, levers_config: list[dict], total_cost: float) -> int:
        num_players = len(players)
        name_to_idx = {p["name"]: idx for idx, p in enumerate(players)}
        names = [p["name"] for p in players]

        pending_speaker_idx = None
        dialogue_pairs = []
        turns_silent = {p["name"]: 0 for p in players}
        last_speaker_name = None

        for b_turn in range(max_turns):
            turn_counter += 1

            # 1. Защита от голодания (starvation prevention):
            # Если кто-то молчит 3 или более раундов, он получает безусловный приоритет на выступление
            max_silent_turns = max(turns_silent.values()) if turns_silent else 0
            starved_player = None
            if max_silent_turns >= 3:
                candidates = [p_name for p_name in names if turns_silent[p_name] == max_silent_turns and p_name != last_speaker_name]
                if candidates:
                    starved_player = candidates[0]

            # 2. Ограничение повтора реплик между одной парой (максимум 2 раза подряд)
            if pending_speaker_idx is not None and len(dialogue_pairs) >= 2:
                last_s, last_t = dialogue_pairs[-1]
                if last_t is not None:
                    target_pair = {last_s, last_t}
                    recent_count = 0
                    for s_name, t_name in reversed(dialogue_pairs):
                        if t_name is not None and {s_name, t_name} == target_pair:
                            recent_count += 1
                        else:
                            break
                    if recent_count >= 2:
                        pending_speaker_idx = None

            # 3. Определение следующего оратора
            if starved_player is not None:
                pending_name = players[pending_speaker_idx]["name"] if pending_speaker_idx is not None else None
                if pending_name is None or turns_silent[starved_player] > turns_silent.get(pending_name, 0):
                    current_idx = name_to_idx[starved_player]
                    pending_speaker_idx = None
                else:
                    current_idx = pending_speaker_idx
                    pending_speaker_idx = None
            elif pending_speaker_idx is not None:
                # Cooldown: один и тот же игрок не может говорить два раунда подряд
                if players[pending_speaker_idx]["name"] == last_speaker_name:
                    eligible = [p_name for p_name in names if p_name != last_speaker_name]
                    silent_longest = max(eligible, key=lambda p_name: (turns_silent[p_name], -name_to_idx[p_name]))
                    current_idx = name_to_idx[silent_longest]
                else:
                    current_idx = pending_speaker_idx
                pending_speaker_idx = None
            else:
                eligible = [p_name for p_name in names if p_name != last_speaker_name]
                if not eligible:
                    eligible = names
                silent_longest = max(eligible, key=lambda p_name: (turns_silent[p_name], -name_to_idx[p_name]))
                current_idx = name_to_idx[silent_longest]

            current = players[current_idx]
            speaker = current["name"]
            last_speaker_name = speaker

            for p_name in names:
                turns_silent[p_name] = 0 if p_name == speaker else turns_silent[p_name] + 1

            prompt = self._build_prompt_phase2(
                speaker=speaker,
                player_prompt=current["prompt"],
                profile=current["profile"],
                state=state,
                history=history,
                rules_prompt=rules,
                turn_in_phase=b_turn + 1,
                max_phase_turns=max_turns,
                global_turn=turn_counter,
                levers_config=levers_config,
                total_cost=total_cost,
            )

            cur_total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
            cur_deficit = max(0.0, total_cost - cur_total_c)
            if cur_deficit > 1000:
                deficit_clause = (
                    f"⚠️ Дефицит бюджета: {format_money(cur_deficit)}. "
                    "Торгуйся, выдвигай встречные требования или предлагай компромиссное финансирование."
                )
            else:
                deficit_clause = f"✅ БЮДЖЕТ 100% СОБРАН ({format_money(total_cost)})! Можно переходить к подтверждению сделки."

            system_message = (
                "Ты участник общественных слушаний по экологическому кризису в городе N. "
                "ИДЕТ ЭТАП 2: ПЕРЕГОВОРЫ И СОГЛАСОВАНИЕ УСЛОВИЙ. "
                f"{deficit_clause} "
                "Веди живой диалог от лица своей роли: заявляй или изменяй свой взнос, переключай доступные условия, "
                "выдвигай встречные требования и ультиматумы к другим участникам. "
                "НЕ повторяй свои вступительные речи слово в слово — отвечай на реплики оппонентов и развивай компромисс! "
                "Обращайся к конкретным сторонам через @Имя или прямо в речи. "
                "Обсуждай параметры текущего проекта: взносы, тарифы, датчики, сроки, мощность, "
                "льготы, мораторий и питьевую воду; не выдумывай внешние структуры и источники средств. "
                "Обязательства и требования фиксируются из структурированных полей ответа."
            )

            prompt += "\n\n" + await self._rag_context(speaker, current["profile"], history, 2)

            answer, action = await self._structured_turn(
                current, "Переговоры", turn_counter,
                [SystemMessage(content=system_message), HumanMessage(content=prompt)],
                state, levers_config, total_cost,
            )
            command = answer.speech if answer is not None else ""
            target_player = action.get("target")
            dialogue_pairs.append((speaker, target_player))
            pending_speaker_idx = name_to_idx.get(target_player)

            history.append({
                "turn": turn_counter,
                "phase": 2,
                "speaker": speaker,
                "target": target_player,
                "command": command,
                "structured_response": answer.model_dump() if answer is not None else None,
                "action": action,
                "state": copy.deepcopy(state),
            })
            self._save_progress()

            # Проверка условий завершения торга и перехода к голосованию (Фаза 3)
            total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
            budget_covered = total_c >= (total_cost - 100_000.0)
            has_pending_demands = any(d.get("status") == "pending" for d in state.get("demands", []))

            # Переговоры не завершаются раньше, чем каждый участник высказался хотя бы дважды
            min_turns_to_finish = num_players * 2

            if (b_turn + 1) >= min_turns_to_finish:
                recent_history = history[-num_players:]

                # 2. Бюджет полностью собран И нет неразрешенных встречных требований
                if budget_covered and not has_pending_demands:
                    break

                # 3. Полный застой в переговорах: за последние раунды никто не менял параметров
                has_any_changes = any(h.get("action", {}).get("applied") for h in recent_history)
                if not has_any_changes and (b_turn + 1) >= (num_players * 3):
                    break

        # Проверяем, нужен ли Last Call (Финальный раунд закрытия дефицита)
        total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
        if total_c < (total_cost - 100_000.0):
            turn_counter = await self._run_last_call(
                players=players,
                state=state,
                history=history,
                rules=rules,
                turn_counter=turn_counter,
                levers_config=levers_config,
                total_cost=total_cost,
            )

        return turn_counter

    async def _run_last_call(self, players: list[dict], state: dict, history: list[dict], rules: str, turn_counter: int, levers_config: list[dict], total_cost: float) -> int:
        for current in players:
            total_c = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
            cur_deficit = max(0.0, total_cost - total_c)
            if cur_deficit <= 1000:
                break

            speaker = current["name"]
            base_budget = current["profile"].get("money", 0)
            p_state = state["players"].get(speaker, {})
            current_contrib = p_state.get("contribution") or 0.0
            spent_levers = p_state.get("spent_on_levers", 0.0)
            available_money = max(0.0, base_budget - spent_levers - current_contrib)

            if base_budget <= 0 and not any(l["owner"] == speaker for l in levers_config):
                continue

            turn_counter += 1
            prompt = self._build_prompt_last_call(
                speaker=speaker,
                player_prompt=current["prompt"],
                profile=current["profile"],
                state=state,
                history=history,
                rules_prompt=rules,
                turn_counter=turn_counter,
                levers_config=levers_config,
                total_cost=total_cost,
                deficit=cur_deficit,
                available_money=available_money,
            )

            system_message = (
                "ВНИМАНИЕ! ИДЕТ ФИНАЛЬНЫЙ РАУНД (LAST CALL) ПЕРЕД СОСТАВЛЕНИЕМ СОГЛАШЕНИЯ И ГОЛОСОВАНИЕМ. "
                f"Текущий реальный дефицит бюджета проекта: {format_money(cur_deficit)}. "
                "Это твой последний шанс подтвердить или увеличить свой взнос, скорректировать параметры или согласовать компромисс. "
                "Если ты не можешь увеличить взнос, подтверди свою окончательную позицию."
            )

            prompt += "\n\n" + await self._rag_context(speaker, current["profile"], history, 2, retrieve=False)

            answer, action = await self._structured_turn(
                current, "Последний раунд", turn_counter,
                [SystemMessage(content=system_message), HumanMessage(content=prompt)],
                state, levers_config, total_cost,
            )
            command = answer.speech if answer is not None else ""
            history.append({
                "turn": turn_counter,
                "phase": 2,
                "is_last_call": True,
                "speaker": speaker,
                "target": action.get("target"),
                "command": command,
                "structured_response": answer.model_dump() if answer is not None else None,
                "action": action,
                "state": copy.deepcopy(state),
            })
            self._save_progress()

        return turn_counter

    async def _run_phase3(self, players: list[dict], state: dict, history: list[dict], turn_counter: int, levers_config: list[dict], total_cost: float) -> Data:
        num_players = len(players)
        total_money = sum(v["contribution"] for v in state["players"].values() if v.get("contribution") is not None)
        deficit = max(0.0, total_cost - total_money)
        majority_threshold = (num_players // 2) + 1

        voting_records = []

        for p_idx in range(num_players):
            turn_counter += 1
            current = players[p_idx]
            speaker = current["name"]

            prompt = self._build_prompt_phase3(
                speaker=speaker,
                player_prompt=current["prompt"],
                profile=current["profile"],
                state=state,
                turn_in_phase=p_idx + 1,
                total_voters=num_players,
                majority_threshold=majority_threshold,
                levers_config=levers_config,
                total_cost=total_cost,
            )

            deficit_info = f"Текущий статус бюджета проекта: {'⚠️ ДЕФИЦИТ ' + format_money(deficit) if deficit > 100_000.0 else '✅ ПОЛНОСТЬЮ ЗАКРЫТ'}. "
            system_message = (
                "ИДЕТ ЭТАП 3: ФИНАЛЬНОЕ ПОИМЕННОЕ ГОЛОСОВАНИЕ ПО ПРОЕКТУ СОГЛАШЕНИЯ. "
                "Оцени зафиксированный Проект Соглашения строго с точки зрения интересов и красных линий своей роли. "
                f"{deficit_info}"
                "Прими взвешенное решение: если компромисс приемлем, допустимо голосовать ЗА или ВОЗДЕРЖАЛСЯ. "
                "Не вспоминай эмоциональные споры из диалога — голосуй исключительно по тексту Соглашения и своему чек-листу! "
                "Верни структурированный голос в vote, обоснование — в speech. "
                f"Соглашение принимается большинством голосов (не менее {majority_threshold} из {num_players}). Дефицит бюджета (при наличии) фиксируется в протоколе для покрытия из резервных фондов."
            )

            prompt += "\n\n" + await self._rag_context(speaker, current["profile"], history, 3, retrieve=False)

            answer, action = await self._structured_turn(
                current, "Голосование", turn_counter,
                [SystemMessage(content=system_message), HumanMessage(content=prompt)],
                state, levers_config, total_cost, voting=True,
            )
            command = answer.speech if answer is not None else ""
            vote_decision = action["vote"]
            vote_comment = action["comment"]

            voting_records.append({
                "player": speaker,
                "vote": vote_decision,
                "comment": vote_comment,
                "status": action["status"],
            })

            history.append({
                "turn": turn_counter,
                "phase": 3,
                "speaker": speaker,
                "target": action.get("target"),
                "command": command,
                "structured_response": answer.model_dump() if answer is not None else None,
                "action": action,
                "state": copy.deepcopy(state),
            })
            self._save_progress()

        votes_for = sum(1 for v in voting_records if v["vote"] == "ЗА")
        votes_against = sum(1 for v in voting_records if v["vote"] == "ПРОТИВ")
        votes_abstain = sum(1 for v in voting_records if v["vote"] == "ВОЗДЕРЖАЛСЯ")
        # Протокол принимается демократическим большинством голосов
        budget_closed = (deficit <= 100_000.0)
        protocol_passed = (votes_for >= majority_threshold)

        voting_summary = {
            "records": voting_records,
            "votes_for": votes_for,
            "votes_against": votes_against,
            "votes_abstain": votes_abstain,
            "votes_invalid": sum(v["status"] == "invalid_response" for v in voting_records),
            "majority_threshold": majority_threshold,
            "quorum": num_players,
            "deficit": deficit,
            "budget_closed": budget_closed,
            "passed": protocol_passed,
        }

        serializable_profiles = []
        for p in players:
            prof = copy.copy(p["profile"])
            prof.pop("llm", None)
            serializable_profiles.append(prof)

        dialogue_text = render_dialogue(
            history=history,
            state=state,
            voting_summary=voting_summary,
            total_cost=total_cost,
            levers_config=levers_config,
        )

        return Data(
            text=dialogue_text,
            data={
                "report": dialogue_text,
                "files": {},
                "history": history,
                "rag_retrievals": getattr(self, "_rag_audit", []),
                "players": serializable_profiles,
                "state": copy.deepcopy(state),
                "voting_summary": voting_summary,
                "consensus_reached": protocol_passed,
                "total_money": total_money,
                "total_cost": total_cost,
                "deficit": deficit,
            },
        )


    async def run_chat(self) -> Data:
        self._rag_audit = []
        self._debug_calls = []
        self._debug_initial_state = None
        self._run_store = None
        history = []
        state = None
        try:
            request_timeout = float(getattr(self, "request_timeout_seconds", 60))
            game_timeout = float(getattr(self, "game_timeout_seconds", 900))
            attempts = int(getattr(self, "request_attempts", 2))
            if not 0 < request_timeout <= 3600 or not 0 < game_timeout <= 86400 or not 1 <= attempts <= 3:
                raise ValueError("Лимит запроса: 1–3600 секунд; игры: 1–86400 секунд; попыток: 1–3")
            players, names, state, settings_data = self._init_session()
            self._debug_initial_state = copy.deepcopy(state)
            rules = str(getattr(self, "rules_prompt", "") or "").strip()
            total_cost = float(settings_data.get("total_cost", DEFAULT_PROJECT_COST))
            levers_config = settings_data.get("levers") or _get_default_settings()["levers"]
            max_turns = max(5, int(settings_data.get("max_rounds") or getattr(self, "turns", 15) or 15))
            self._live_players, self._live_history = players, history
            self._live_state, self._live_cost, self._live_levers = state, total_cost, levers_config
            # Сохранение запусков на диск отключено.
            async with asyncio.timeout(game_timeout):
                turn_counter = await self._run_phase1(players, state, history, rules, 0, levers_config, total_cost)
                turn_counter = await self._run_phase2(players, state, history, rules, turn_counter, max_turns, levers_config, total_cost)
                result = await self._run_phase3(players, state, history, turn_counter, levers_config, total_cost)
            result.data["status"] = "completed"
            result.data["error"] = None
            result.data["debug"] = self._debug_data(history, state)
            self.status = "Игра завершена"
            return result
        except asyncio.CancelledError:
            # LF отменяет coroutine; не подавляем отмену и не запускаем следующие ходы.
            # При отмене графа downstream не выполнится, поэтому отчёт сохраняем локально.
            try:
                self._save_progress("cancelled", "Выполнение отменено в Langflow")
            except OSError:
                pass  # Уже сохранённые ходы остаются на диске; отмена имеет приоритет.
            raise
        except Exception as exc:
            detail = str(exc)
            if isinstance(exc, TimeoutError) and not detail:
                detail = "Истёк общий лимит времени игры"
            error = f"{type(exc).__name__}: {detail}"
            self.status = "Игра прервана: " + type(exc).__name__
            if self._run_store is not None:
                try:
                    return self._save_progress("failed", error)
                except OSError as save_error:
                    error += f"; не удалось обновить файлы: {type(save_error).__name__}"
            # Ошибка конфигурации/файловой системы тоже должна дойти до ReportController.
            text = "# Игра прервана\n\n" + error
            return Data(data={"status": "failed", "error": error, "report": text, "files": {},
                              "history": history, "state": state, "consensus_reached": None,
                              "debug": self._debug_data(history, state)}, text=text)
