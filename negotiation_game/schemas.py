"""Структурированные ответы игроков; не зависят от Langflow и LangChain."""

import copy
import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
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
