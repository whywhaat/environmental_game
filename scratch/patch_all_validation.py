# -*- coding: utf-8 -*-
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update _clean_speech_message
old_clean = """        # Удаляем артефакты chunk_id
        line_s = re.sub(r"\\[chunk_id[^\\]]*\\]", "", line_s)
        # Удаляем остатки команд в круглых и квадратных скобках, например [SET ...] или (SET ...)
        line_s = re.sub(r"[\\[\\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\\]\\)]*[\\]\\)]", "", line_s, flags=re.I).strip()"""

new_clean = """        # Удаляем любые артефакты чанков RAG [chunk...] и [chunk_id...]
        line_s = re.sub(r"\\[chunk[^\\]]*\\]", "", line_s, flags=re.I)
        # Удаляем остатки команд в круглых и квадратных скобках, например [SET ...] или (SET ...)
        line_s = re.sub(r"[\\[\\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\\]\\)]*[\\]\\)]", "", line_s, flags=re.I).strip()
        # Удаляем незамкнутые скобки команд в конце строк, например: [SET me Взнос 160 млн;
        line_s = re.sub(r"[\\[\\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\\]\\n]*$", "", line_s, flags=re.I).strip()
        # Удаляем одиночные остаточные команды SET/DEMAND в конце строк
        line_s = re.sub(r"\\b(?:SET|DEMAND|ТРЕБУЮ)\\s+(?:me|@[A-Za-zА-Яа-я0-9_]+)[^;\\n]*;?$", "", line_s, flags=re.I).strip()"""

# 2. Update _normalize_lever_value for plant_mode & None handling
old_norm = """        elif l_type == "str":
            low = str(raw_val).lower().strip()
            if any(w in low for w in ["shared", "обоюд", "паритет", "компромисс"]):
                return "shared"
            elif any(w in low for w in ["holding", "завод", "холдинг", "комбинат"]):
                return "holding"
            elif any(w in low for w in ["city", "город", "администр"]):
                return "city"
            elif "70" in low:
                return "70%"
            elif "100" in low or "полн" in low:
                return "100%"
            return str(raw_val).strip(" '\\"")"""

new_norm = """        elif l_type == "str":
            if raw_val is None or str(raw_val).strip().lower() in {"none", "null", ""}:
                return lev.get("default", "100%") if lev else "100%"
            low = str(raw_val).lower().strip()
            if any(w in low for w in ["shared", "обоюд", "паритет", "компромисс"]):
                return "shared"
            elif any(w in low for w in ["holding", "завод", "холдинг", "комбинат"]):
                return "holding"
            elif any(w in low for w in ["city", "город", "администр"]):
                return "city"
            elif any(w in low for w in ["70", "семьдесят"]):
                return "70%"
            elif any(w in low for w in ["100", "полн", "штатн", "стандарт", "обычн", "неприемлем", "отказ", "не соглас"]):
                return "100%"
            return str(raw_val).strip(" '\\"")"""

# 3. Update _init_session: store max_budget in state["players"]
old_init_state = """        state = {
            "total_cost": total_cost,
            "players": {
                name: {
                    "contribution": None,
                    "share": None,
                    "spent_on_levers": 0.0,
                }
                for name in names
            },"""

new_init_state = """        state = {
            "total_cost": total_cost,
            "players": {
                p["name"]: {
                    "contribution": None,
                    "share": None,
                    "spent_on_levers": 0.0,
                    "max_budget": float(p["profile"].get("money", 0.0)),
                }
                for p in players
            },"""

# 4. Add _validate_and_sanitize_actions method and call it in _apply_actions
old_apply_header = """    @staticmethod
    def _apply_actions(state: dict, parsed: dict, target: str | None, total_cost: float, levers_config: list[dict] | None = None) -> dict:
        \"\"\"Применяет накопленные действия (деньги, крутилки, требования) к состоянию игры.\"\"\"
        applied_log = []
        speaker = parsed.get("speaker") or list(state.get("players", {}).keys())[0]
        all_names = list(state.get("players", {}).keys())

        for act in parsed.get("actions", []):"""

new_apply_header = """    @staticmethod
    def _validate_and_sanitize_actions(
        actions: list[dict],
        speaker: str,
        state: dict,
        levers_config: list[dict],
        total_cost: float,
    ) -> list[dict]:
        \"\"\"
        Алгоритмический валидатор действий:
        1. Проверка бюджетов: игрок не может внести больше своего максимального бюджета.
        2. Финансовые требования к другим сторонам не могут превышать бюджет целевой стороны (если у нее 0, требование отбрасывается).
        3. Полномочия по рычагам: игрок может устанавливать (LEVER_SET) только свои рычаги.
        4. Чужие рычаги автоматически конвертируются в требования (DEMAND_SET) строго к владельцу рычага.
        5. Требования к самому себе отбрасываются или конвертируются в прямое действие.
        6. Значения рычагов строго валидируются и нормализуются по типу и допустимым значениям (исключая None/'None').
        \"\"\"
        all_names = list(state.get("players", {}).keys())
        clean_actions = []

        for act in actions:
            act_type = act.get("type")

            # 1. ВЗНОС СТОРОНЫ
            if act_type == "PLEDGE_SET":
                raw_player = act.get("player") or speaker
                resolved_p = _resolve_player_name(raw_player, all_names) or speaker
                p_state = state.get("players", {}).get(resolved_p, {})
                max_b = p_state.get("max_budget")

                try:
                    val = float(act.get("value", 0.0))
                except (ValueError, TypeError):
                    continue

                if 0 < val <= 1000 and total_cost > 1000:
                    val *= 1_000_000.0

                # Сторона с нулевым бюджетом не может делать взносы
                if max_b is not None and max_b <= 0:
                    continue

                # Если взнос превышает бюджет стороны (например, Администрация цитирует 160 млн холдинга):
                if max_b is not None and val > max_b:
                    if val > max_b * 1.1:
                        # Очевидная ссылка на чужой взнос -> отбрасываем
                        continue
                    else:
                        val = max_b

                if val < 0:
                    continue

                clean_actions.append({
                    "type": "PLEDGE_SET",
                    "player": resolved_p,
                    "field": act.get("field", "contribution"),
                    "value": val,
                    "text": act.get("text", f"Взнос {resolved_p}: {_format_money(val)}"),
                })

            # 2. ТРЕБОВАНИЕ ВЗНОСА К ДРУГОЙ СТОРОНЕ
            elif act_type == "DEMAND_PLEDGE":
                from_p = act.get("from") or speaker
                raw_target = act.get("target") or "все"
                resolved_t = _resolve_player_name(raw_target, all_names) or str(raw_target).lstrip("@")

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

                # Нельзя требовать деньги от стороны с 0 бюджетом (например, от Эко-НКО или Жителей)
                if target_max_b is not None and target_max_b <= 0:
                    continue

                # Ограничиваем требование максимальной вместимостью бюджета целевой стороны
                if target_max_b is not None and val > target_max_b:
                    val = target_max_b

                if val <= 0:
                    continue

                clean_actions.append({
                    "type": "DEMAND_PLEDGE",
                    "demand_type": "pledge",
                    "from": from_p,
                    "target": resolved_t,
                    "field": "contribution",
                    "value": val,
                    "text": f"Взнос {_format_money(val)}",
                })

            # 3. ПЕРЕКЛЮЧЕНИЕ РЫЧАГА
            elif act_type == "LEVER_SET":
                l_id = act.get("lever_id")
                lev = TwoLLMContractGame._find_lever(str(l_id), levers_config)
                if not lev:
                    continue

                norm_val = TwoLLMContractGame._normalize_lever_value(lev, act.get("value"))
                if norm_val is None or str(norm_val).strip().lower() in {"none", "null", ""}:
                    norm_val = lev.get("default")

                if _is_owner(lev["owner"], speaker):
                    cost = lev.get("cost", 0.0) if (norm_val is True or (isinstance(norm_val, (int, float)) and norm_val > 0)) else 0.0
                    clean_actions.append({
                        "type": "LEVER_SET",
                        "lever_id": lev["id"],
                        "title": lev["title"],
                        "value": norm_val,
                        "cost": cost,
                        "owner": lev["owner"],
                        "player": speaker,
                    })
                else:
                    clean_actions.append({
                        "type": "DEMAND_SET",
                        "demand_type": "lever",
                        "from": speaker,
                        "target": lev["owner"],
                        "lever_id": lev["id"],
                        "title": lev["title"],
                        "value": norm_val,
                        "text": f"{lev['title']}: {norm_val}",
                    })

            # 4. ТРЕБОВАНИЕ ПО РЫЧАГУ
            elif act_type == "DEMAND_SET":
                from_p = act.get("from") or speaker
                l_id = act.get("lever_id") or act.get("title")
                lev = TwoLLMContractGame._find_lever(str(l_id), levers_config)
                if not lev:
                    continue

                norm_val = TwoLLMContractGame._normalize_lever_value(lev, act.get("value"))
                if norm_val is None or str(norm_val).strip().lower() in {"none", "null", ""}:
                    norm_val = lev.get("default")

                if _is_owner(lev["owner"], from_p):
                    cost = lev.get("cost", 0.0) if (norm_val is True or (isinstance(norm_val, (int, float)) and norm_val > 0)) else 0.0
                    clean_actions.append({
                        "type": "LEVER_SET",
                        "lever_id": lev["id"],
                        "title": lev["title"],
                        "value": norm_val,
                        "cost": cost,
                        "owner": lev["owner"],
                        "player": from_p,
                    })
                else:
                    clean_actions.append({
                        "type": "DEMAND_SET",
                        "demand_type": "lever",
                        "from": from_p,
                        "target": lev["owner"],
                        "lever_id": lev["id"],
                        "title": lev["title"],
                        "value": norm_val,
                        "text": f"{lev['title']} ({norm_val})",
                    })

            elif act_type == "DEMAND_WITHDRAW":
                clean_actions.append(act)

        return clean_actions

    @staticmethod
    def _apply_actions(state: dict, parsed: dict, target: str | None, total_cost: float, levers_config: list[dict] | None = None) -> dict:
        \"\"\"Применяет накопленные действия (деньги, крутилки, требования) к состоянию игры.\"\"\"
        applied_log = []
        speaker = parsed.get("speaker") or list(state.get("players", {}).keys())[0]
        all_names = list(state.get("players", {}).keys())
        cfg = levers_config or []

        # Алгоритмическая валидация и нормализация всех действий
        validated_actions = TwoLLMContractGame._validate_and_sanitize_actions(
            actions=parsed.get("actions", []),
            speaker=speaker,
            state=state,
            levers_config=cfg,
            total_cost=total_cost,
        )

        for act in validated_actions:"""

# 5. Update _build_parser_prompt
old_parser_prompt = """        owned_levers = [lev for lev in levers_config if _is_owner(lev.get("owner", ""), speaker)]
        owned_str = ", ".join(f"{l['title']} (ID: {l['id']})" for l in owned_levers) if owned_levers else "нет рычагов в прямом ведении"

        all_players_str = ", ".join(state.get("players", {}).keys())"""

new_parser_prompt = """        owned_levers = [lev for lev in levers_config if _is_owner(lev.get("owner", ""), speaker)]
        owned_str = ", ".join(f"{l['title']} (ID: {l['id']})" for l in owned_levers) if owned_levers else "нет рычагов в прямом ведении"

        sp_max_b = sp_data.get("max_budget", 0.0)
        sp_budget_str = _format_money(sp_max_b)

        all_players_str = ", ".join(state.get("players", {}).keys())"""

replacements = [
    (old_clean, new_clean, "1. _clean_speech_message"),
    (old_norm, new_norm, "2. _normalize_lever_value"),
    (old_init_state, new_init_state, "3. _init_session max_budget"),
    (old_apply_header, new_apply_header, "4. _validate_and_sanitize_actions"),
    (old_parser_prompt, new_parser_prompt, "5. _build_parser_prompt budget"),
]

for old_s, new_s, desc in replacements:
    if old_s in code:
        code = code.replace(old_s, new_s, 1)
        print(f"OK: Replaced {desc}")
    else:
        old_crlf = old_s.replace("\n", "\r\n")
        new_crlf = new_s.replace("\n", "\r\n")
        if old_crlf in code:
            code = code.replace(old_crlf, new_crlf, 1)
            print(f"OK: Replaced {desc} (CRLF)")
        else:
            print(f"ERROR: Could not find block for: {desc}")
            exit(1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

py_compile.compile(file_path, doraise=True)
print("dialog.py compiled successfully with py_compile!")
