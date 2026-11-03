"""Проверка прав и атомарная подготовка хода без LLM и Langflow.

prepare_player_turn возвращает новое состояние и действие, не изменяя входное.
"""

import copy
import math

from .schemas import PlayerTurn


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

