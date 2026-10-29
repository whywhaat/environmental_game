import sys

path = "langflow/custom_components/my_components/dialog.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update _clean_speech_message
old_clean_speech = '''def _clean_speech_message(msg: str) -> str:
    if not msg:
        return ""
    # Удаление markdown-блоков кода
    res = re.sub(r"```[a-zA-Z]*\\n?.*?\\n?```", "", msg, flags=re.S)
    # Удаление системных баннеров === ... ===
    res = re.sub(r"===\\s*.+?\\s*===", "", res, flags=re.S)

    # Построчная очистка'''

new_clean_speech = '''def _clean_speech_message(msg: str) -> str:
    if not msg:
        return ""
    # Удаление markdown-блоков кода
    res = re.sub(r"```[a-zA-Z]*\\n?.*?\\n?```", "", msg, flags=re.S)
    # Удаление системных баннеров === ... ===
    res = re.sub(r"===\\s*.+?\\s*===", "", res, flags=re.S)
    # Удаление любых артефактов чанков RAG [chunk...] и [chunk_id...]
    res = re.sub(r"\\[\\s*chunk[^\\]]*\\]", "", res, flags=re.I)
    # Удаление тегов <speech> и </speech>
    res = re.sub(r"</?speech>", "", res, flags=re.I)

    # Построчная очистка'''

assert old_clean_speech in code, "old_clean_speech not found!"
code = code.replace(old_clean_speech, new_clean_speech, 1)

# Inside _clean_speech_message: add prompt leak filter
old_prompt_filter_anchor = '''        # Пропускаем служебные строки команд (включая команды в скобках [SET ...] или (SET ...))
        if re.match(r"^[\\s*+\\-]*[\\[\\(]?\\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\\b", line_s, flags=re.I):
            continue'''

new_prompt_filter_anchor = '''        # Пропускаем служебные строки команд (включая команды в скобках [SET ...] или (SET ...))
        if re.match(r"^[\\s*+\\-]*[\\[\\(]?\\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\\b", line_s, flags=re.I):
            continue
        # Фильтруем утечки системных инструкций и промптов парсера
        line_s = re.sub(r"(?i)Верни\\s+СТРОГО\\s+валидный\\s+JSON.*", "", line_s).strip()
        line_s = re.sub(r"(?i)структуры\\s+без\\s+markdown-оберток.*", "", line_s).strip()
        line_s = re.sub(r"(?i)Реплика\\s+игрока\\s*:?", "", line_s).strip()
        line_s = re.sub(r"(?i)ВЫСТУПЛЕНИЕ\\s+УЧАСТНИКА\\s*:?", "", line_s).strip()'''

assert old_prompt_filter_anchor in code, "old_prompt_filter_anchor not found!"
code = code.replace(old_prompt_filter_anchor, new_prompt_filter_anchor, 1)

# 2. Update human_prompt in _build_parser_prompt
old_human_prompt = '''            f"ВЫСТУПЛЕНИЕ УЧАСТНИКА:\\n{text}\\n\\n"
            "Верни СТРОГО валидный JSON следующей структуры без markdown-оберток:\\n"
            "{\\n"
            '  "own_contribution": <число_млн_или_null>,\\n'
            '  "own_levers": {"<id_рычага>": <значение>},\\n'
            '  "demands": [\\n'
            '    {\\n'
            '      "target": "<имя_или_роль_стороны>",\\n'
            '      "demand_type": "pledge" или "lever",\\n'
            '      "lever_id": "<id_рычага_или_null>",\\n'
            '      "value": <число_млн_или_значение_рычага>,\\n'
            '      "text": "<краткая суть требования/ультиматума>"\\n'
            '    }\\n'
            '  ],\\n'
            '  "withdrawn_demands": ["<имя_стороны_или_рычаг_или_all>"],\\n'
            '  "addressed_to": "<имя_стороны_или_null>",\\n'
            '  "clean_speech": "<очищенная речь участника для стенограммы>"\\n'
            "}"'''

new_human_prompt = '''            f"ВЫСТУПЛЕНИЕ УЧАСТНИКА:\\n<speech>\\n{text}\\n</speech>\\n\\n"
            "Верни СТРОГО валидный JSON следующей структуры без markdown-оберток:\\n"
            "{\\n"
            '  "own_contribution": <число_млн_или_null>,\\n'
            '  "own_levers": {"<id_рычага>": <значение>},\\n'
            '  "demands": [\\n'
            '    {\\n'
            '      "target": "<имя_или_роль_стороны>",\\n'
            '      "demand_type": "pledge" или "lever",\\n'
            '      "lever_id": "<id_рычага_или_null>",\\n'
            '      "value": <число_млн_или_значение_рычага>,\\n'
            '      "text": "<краткая суть требования/ультиматума>"\\n'
            '    }\\n'
            '  ],\\n'
            '  "withdrawn_demands": ["<имя_стороны_или_рычаг_или_all>"],\\n'
            '  "addressed_to": "<имя_стороны_или_null>",\\n'
            '  "clean_speech": "<очищенная прямая речь участника строго из блока <speech> для стенограммы, без технических команд и без служебных инструкций>"\\n'
            "}"'''

assert old_human_prompt in code, "old_human_prompt not found!"
code = code.replace(old_human_prompt, new_human_prompt, 1)

# 3. Update _validate_and_sanitize_actions target resolution
old_target_res = '''            elif act_type == "DEMAND_PLEDGE":
                from_p = act.get("from") or speaker
                raw_target = act.get("target") or "все"
                resolved_t = _resolve_player_name(raw_target, all_names) or str(raw_target).lstrip("@")'''

new_target_res = '''            elif act_type == "DEMAND_PLEDGE":
                from_p = act.get("from") or speaker
                raw_target = str(act.get("target") or "все").lstrip("@").strip()
                resolved_t = _resolve_player_name(raw_target, all_names) or raw_target'''

assert old_target_res in code, "old_target_res not found!"
code = code.replace(old_target_res, new_target_res, 1)

# 4. Update _render_voter_checklist lever status rendering
old_lev_status = '''        current_levers = state.get("levers", {})
        for lev in levers_config:
            val = current_levers.get(lev["id"], lev.get("default"))
            val_str = "УТВЕРЖДЕНО ✅" if (val is True or (isinstance(val, (int, float)) and val > 0)) else "НЕ ПРИНЯТО ❌"
            if isinstance(val, (int, float)) and lev["type"] == "int":
                val_str = f"УТВЕРЖДЕНО ({val} мес.) ✅"
            lines.append(f"   - {lev['title']}: {val_str}")'''

new_lev_status = '''        current_levers = state.get("levers", {})
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
            lines.append(f"   - {lev['title']}: {val_str}")'''

assert old_lev_status in code, "old_lev_status not found!"
code = code.replace(old_lev_status, new_lev_status, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("All 4 fixes applied successfully to dialog.py!")
