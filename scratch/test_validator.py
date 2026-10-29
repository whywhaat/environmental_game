# -*- coding: utf-8 -*-
import sys

# Test cases for the algorithmic validator
levers_config = [
    {
        "id": "guilt",
        "title": "Признание вины за аварию",
        "type": "str",
        "allowed": ["shared", "holding", "city"],
        "owner": "Регион",
        "default": "shared",
    },
    {
        "id": "plant_mode",
        "title": "Режим работы предприятия",
        "type": "str",
        "allowed": ["100%", "70%"],
        "owner": "Промкомплекс",
        "default": "100%",
    },
    {
        "id": "timeline",
        "title": "Срок реконструкции (мес.)",
        "type": "int",
        "owner": "Промкомплекс",
        "default": 36,
        "min_value": 24,
        "max_value": 48,
    },
    {
        "id": "tariffs_freeze",
        "title": "Заморозка тарифов ЖКХ",
        "type": "bool",
        "owner": "Администрация",
        "default": False,
    },
    {
        "id": "open_sensors",
        "title": "Открытые эко-датчики 24/7",
        "type": "bool",
        "owner": "Эко-НКО",
        "default": False,
    },
]

state = {
    "players": {
        "Администрация": {"contribution": 0.0, "max_budget": 30_000_000.0},
        "Промкомплекс": {"contribution": 160_000_000.0, "max_budget": 180_000_000.0},
        "Регион": {"contribution": 90_000_000.0, "max_budget": 90_000_000.0},
        "Жители": {"contribution": 0.0, "max_budget": 0.0},
        "Эко-НКО": {"contribution": 0.0, "max_budget": 0.0},
    },
    "levers": {},
    "demands": []
}

def _is_owner(owner, speaker):
    o = owner.lower()
    s = speaker.lower()
    return o in s or s in o

def _normalize_lever_value(lev, raw_val):
    l_type = lev.get("type", "bool") if lev else "bool"
    if raw_val is None or str(raw_val).strip().lower() in {"none", "null", ""}:
        return lev.get("default", "100%") if lev else "100%"
    if l_type == "bool":
        if isinstance(raw_val, bool): return raw_val
        s = str(raw_val).lower().strip()
        return any(w in s for w in ["да", "yes", "true", "1", "вкл", "+"])
    elif l_type == "int":
        import re
        d = re.search(r"\d+", str(raw_val))
        return int(d.group(0)) if d else lev.get("default", 36)
    elif l_type == "str":
        low = str(raw_val).lower().strip()
        if any(w in low for w in ["shared", "обоюд"]): return "shared"
        if any(w in low for w in ["holding", "завод"]): return "holding"
        if any(w in low for w in ["70", "семьдесят"]): return "70%"
        if any(w in low for w in ["100", "штатн", "стандарт", "неприемлем"]): return "100%"
        return str(raw_val).strip()
    return raw_val

def validate_actions(actions, speaker, state, levers_config, total_cost=270_000_000.0):
    all_names = list(state.get("players", {}).keys())
    clean_actions = []

    for act in actions:
        act_type = act.get("type")

        # 1. PLEDGE_SET
        if act_type == "PLEDGE_SET":
            raw_player = act.get("player") or speaker
            resolved_p = raw_player
            p_state = state.get("players", {}).get(resolved_p, {})
            max_b = p_state.get("max_budget")

            try:
                val = float(act.get("value", 0.0))
            except (ValueError, TypeError):
                continue
            if 0 < val <= 1000 and total_cost > 1000:
                val *= 1_000_000.0

            if max_b is not None:
                if max_b <= 0:
                    continue  # zero budget role cannot pledge
                if val > max_b:
                    if val > max_b * 1.1:
                        # Obvious misattribution quoting another player's contribution
                        continue
                    else:
                        val = max_b

            clean_actions.append({"type": "PLEDGE_SET", "player": resolved_p, "value": val})

        # 2. DEMAND_PLEDGE
        elif act_type == "DEMAND_PLEDGE":
            from_p = act.get("from") or speaker
            resolved_t = act.get("target")

            if _is_owner(resolved_t, from_p):
                continue

            try:
                val = float(act.get("value", 0.0))
            except (ValueError, TypeError):
                continue
            if 0 < val <= 1000 and total_cost > 1000:
                val *= 1_000_000.0

            target_state = state.get("players", {}).get(resolved_t, {})
            target_max_b = target_state.get("max_budget")

            if target_max_b is not None and target_max_b <= 0:
                continue  # cannot demand money from 0-budget role

            if target_max_b is not None and val > target_max_b:
                val = target_max_b  # cap at target's max budget

            clean_actions.append({"type": "DEMAND_PLEDGE", "from": from_p, "target": resolved_t, "value": val})

        # 3. LEVER_SET
        elif act_type == "LEVER_SET":
            l_id = act.get("lever_id")
            lev = next((l for l in levers_config if l["id"] == l_id), None)
            if not lev: continue
            norm_val = _normalize_lever_value(lev, act.get("value"))

            if _is_owner(lev["owner"], speaker):
                clean_actions.append({"type": "LEVER_SET", "lever_id": lev["id"], "value": norm_val, "player": speaker})
            else:
                clean_actions.append({"type": "DEMAND_SET", "from": speaker, "target": lev["owner"], "lever_id": lev["id"], "value": norm_val})

        # 4. DEMAND_SET
        elif act_type == "DEMAND_SET":
            from_p = act.get("from") or speaker
            l_id = act.get("lever_id")
            lev = next((l for l in levers_config if l["id"] == l_id), None)
            if not lev: continue
            norm_val = _normalize_lever_value(lev, act.get("value"))

            if _is_owner(lev["owner"], from_p):
                clean_actions.append({"type": "LEVER_SET", "lever_id": lev["id"], "value": norm_val, "player": from_p})
            else:
                clean_actions.append({"type": "DEMAND_SET", "from": from_p, "target": lev["owner"], "lever_id": lev["id"], "value": norm_val})

    return clean_actions

print("=== Test 1: Admin quoting 160m of holding ===")
parsed_turn_19 = [
    {"type": "PLEDGE_SET", "player": "Администрация", "value": 160_000_000.0},
    {"type": "LEVER_SET", "lever_id": "tariffs_freeze", "value": True},
]
res1 = validate_actions(parsed_turn_19, "Администрация", state, levers_config)
print("Result 1:", res1)
assert len([a for a in res1 if a["type"] == "PLEDGE_SET"]) == 0, "Admin 160m pledge should be DROPPED!"
assert len([a for a in res1 if a["type"] == "LEVER_SET"]) == 1, "Tariffs freeze should be KEPT!"

print("\n=== Test 2: Residents demanding 160m from Eco-NGO ===")
parsed_turn_20 = [
    {"type": "DEMAND_PLEDGE", "from": "Жители", "target": "Эко-НКО", "value": 160_000_000.0},
    {"type": "DEMAND_SET", "from": "Жители", "target": "Промкомплекс", "lever_id": "timeline", "value": 30},
]
res2 = validate_actions(parsed_turn_20, "Жители", state, levers_config)
print("Result 2:", res2)
assert len([a for a in res2 if a["type"] == "DEMAND_PLEDGE"]) == 0, "160m demand to Eco-NGO should be DROPPED!"
assert len([a for a in res2 if a["type"] == "DEMAND_SET"]) == 1, "Timeline demand should be KEPT!"

print("\n=== Test 3: Plant mode raw None becomes 100% ===")
parsed_turn_1 = [
    {"type": "LEVER_SET", "lever_id": "plant_mode", "value": None},
]
res3 = validate_actions(parsed_turn_1, "Промкомплекс", state, levers_config)
print("Result 3:", res3)
assert res3[0]["value"] == "100%", "None should become '100%'!"

print("\n🎉 ALL VALIDATOR TESTS PASSED!")
