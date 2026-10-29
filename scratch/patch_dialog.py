# -*- coding: utf-8 -*-
import sys
import re
import py_compile

file_path = "langflow/custom_components/my_components/dialog.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = []

# Note: Blocks 1-8 were already applied in the previous run.
# Let's verify if block 1 was already applied:
if 'if re.match(r"^[\s*+\-]*[\[\(]?\b(?:SET' in content:
    print("Blocks 1-8 were already applied successfully!")
else:
    # 1. _clean_speech_message
    old_clean1 = r"""        # Пропускаем служебные строки команд
        if re.match(r"^[\s*+\-]*\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\b", line_s, flags=re.I):
            continue"""
    new_clean1 = r"""        # Пропускаем служебные строки команд (включая команды в скобках [SET ...] или (SET ...))
        if re.match(r"^[\s*+\-]*[\[\(]?\b(?:SET|DEMAND|ТРЕБУЮ|TO|ПЕРЕДАТЬ|РЕШЕНИЕ|VOTE|ГОЛОСУЮ)\b", line_s, flags=re.I):
            continue"""
    replacements.append((old_clean1, new_clean1, "1. _clean_speech_message line starter"))

    old_clean2 = r"""        # Удаляем остатки команд в скобках, например (SET ...) или (DEMAND ...)
        line_s = re.sub(r"\((?:SET|DEMAND|ТРЕБУЮ)[^\)]*\)", "", line_s, flags=re.I).strip()"""
    new_clean2 = r"""        # Удаляем остатки команд в круглых и квадратных скобках, например [SET ...] или (SET ...)
        line_s = re.sub(r"[\[\(](?:SET|DEMAND|ТРЕБУЮ|VOTE|ГОЛОСУЮ)[^\]\)]*[\]\)]", "", line_s, flags=re.I).strip()"""
    replacements.append((old_clean2, new_clean2, "2. _clean_speech_message bracketed commands"))

    # 2. _parse_single_set
    old_parse_single = r"""                if _is_owner(owner, speaker):
                    return {
                        "type": "LEVER_SET",
                        "lever_id": target_lever["id"],
                        "title": target_lever["title"],
                        "value": val,
                        "cost": target_lever.get("cost", 0.0) if (val is True or (isinstance(val, (int, float)) and val > 0)) else 0.0,
                        "owner": owner,
                        "player": speaker,
                    }
                else:
                    return {
                        "type": "DEMAND_SET",
                        "demand_type": "lever",
                        "from": speaker,
                        "target": target_override or owner,
                        "lever_id": target_lever["id"],
                        "title": target_lever["title"],
                        "value": val,
                        "raw_val": raw_val,
                    }"""
    new_parse_single = r"""                if _is_owner(owner, speaker):
                    return {
                        "type": "LEVER_SET",
                        "lever_id": target_lever["id"],
                        "title": target_lever["title"],
                        "value": val,
                        "cost": target_lever.get("cost", 0.0) if (val is True or (isinstance(val, (int, float)) and val > 0)) else 0.0,
                        "owner": owner,
                        "player": speaker,
                    }
                else:
                    final_target = target_override if (target_override and not _is_owner(target_override, speaker)) else owner
                    return {
                        "type": "DEMAND_SET",
                        "demand_type": "lever",
                        "from": speaker,
                        "target": final_target,
                        "lever_id": target_lever["id"],
                        "title": target_lever["title"],
                        "value": val,
                        "raw_val": raw_val,
                    }"""
    replacements.append((old_parse_single, new_parse_single, "3. _parse_single_set self-demand target fix"))

    # 3. _parse_with_llm
    old_llm_demands = r"""                if d_type == "pledge":
                    try:
                        p_val = float(d.get("value", 0.0))
                        if 0 < p_val <= 1000 and total_cost > 1000:
                            p_val = p_val * 1_000_000.0
                        actions.append({
                            "type": "DEMAND_PLEDGE",
                            "demand_type": "pledge",
                            "from": speaker,
                            "target": resolved_target,
                            "field": "contribution",
                            "value": p_val,
                            "text": d_text or f"Взнос {_format_money(p_val)}",
                        })
                    except (ValueError, TypeError):
                        pass
                else:
                    l_id = d.get("lever_id") or ""
                    lev = TwoLLMContractGame._find_lever(str(l_id), levers_config)
                    if not lev and d_text:
                        lev = TwoLLMContractGame._find_lever(d_text, levers_config)
                    title = lev["title"] if lev else (l_id or d_text or "Параметр")
                    target_to_use = resolved_target if resolved_target != "все" else (lev["owner"] if lev else "все")
                    val_norm = TwoLLMContractGame._normalize_lever_value(lev, d.get("value")) if lev else d.get("value")

                    actions.append({
                        "type": "DEMAND_SET",
                        "demand_type": "lever",
                        "from": speaker,
                        "target": target_to_use,
                        "lever_id": lev["id"] if lev else str(l_id),
                        "title": title,
                        "value": val_norm,
                        "text": d_text or f"{title}: {val_norm}",
                    })"""

    new_llm_demands = r"""                if d_type == "pledge":
                    try:
                        p_val = float(d.get("value", 0.0))
                        if 0 < p_val <= 1000 and total_cost > 1000:
                            p_val = p_val * 1_000_000.0
                        if resolved_target and not _is_owner(resolved_target, speaker):
                            actions.append({
                                "type": "DEMAND_PLEDGE",
                                "demand_type": "pledge",
                                "from": speaker,
                                "target": resolved_target,
                                "field": "contribution",
                                "value": p_val,
                                "text": d_text or f"Взнос {_format_money(p_val)}",
                            })
                    except (ValueError, TypeError):
                        pass
                else:
                    l_id = d.get("lever_id") or ""
                    lev = TwoLLMContractGame._find_lever(str(l_id), levers_config)
                    if not lev and d_text:
                        lev = TwoLLMContractGame._find_lever(d_text, levers_config)
                    title = lev["title"] if lev else (l_id or d_text or "Параметр")
                    val_norm = TwoLLMContractGame._normalize_lever_value(lev, d.get("value")) if lev else d.get("value")

                    if lev and _is_owner(lev["owner"], speaker):
                        actions.append({
                            "type": "LEVER_SET",
                            "lever_id": lev["id"],
                            "title": lev["title"],
                            "value": val_norm,
                            "cost": lev.get("cost", 0.0) if (val_norm is True or (isinstance(val_norm, (int, float)) and val_norm > 0)) else 0.0,
                            "owner": lev["owner"],
                            "player": speaker,
                        })
                    else:
                        target_to_use = resolved_target if (resolved_target and resolved_target != "все" and not _is_owner(resolved_target, speaker)) else (lev["owner"] if lev else "все")
                        if not _is_owner(target_to_use, speaker):
                            actions.append({
                                "type": "DEMAND_SET",
                                "demand_type": "lever",
                                "from": speaker,
                                "target": target_to_use,
                                "lever_id": lev["id"] if lev else str(l_id),
                                "title": title,
                                "value": val_norm,
                                "text": d_text or f"{title}: {val_norm}",
                            })"""
    replacements.append((old_llm_demands, new_llm_demands, "4. _parse_with_llm route self-demands to LEVER_SET or owner"))

    # 4. _apply_actions
    old_apply = r"""            elif act_type == "DEMAND_SET":
                from_p = act.get("from", speaker)
                target_p = act.get("target", "")
                l_id = act.get("lever_id")
                title = act.get("title", l_id)
                val = act.get("value")

                demands = state.setdefault("demands", [])"""

    new_apply = r"""            elif act_type == "DEMAND_SET":
                from_p = act.get("from", speaker)
                target_p = act.get("target", "")
                l_id = act.get("lever_id")
                title = act.get("title", l_id)
                val = act.get("value")

                if _is_owner(target_p, from_p):
                    lev = TwoLLMContractGame._find_lever(l_id or title, levers_config)
                    if lev and _is_owner(lev["owner"], from_p):
                        cost = lev.get("cost", 0.0) if (val is True or (isinstance(val, (int, float)) and val > 0)) else 0.0
                        state.setdefault("levers", {})[lev["id"]] = {
                            "value": val,
                            "owner": lev["owner"],
                            "cost": cost,
                            "title": lev["title"],
                            "set_by": from_p,
                        }
                        applied_log.append({
                            "kind": "LEVER",
                            "lever_id": lev["id"],
                            "title": lev["title"],
                            "value": val,
                            "cost": cost,
                            "owner": lev["owner"],
                        })
                        continue
                    elif lev:
                        target_p = lev["owner"]
                    else:
                        continue
                if _is_owner(target_p, from_p):
                    continue

                demands = state.setdefault("demands", [])"""
    replacements.append((old_apply, new_apply, "5. _apply_actions intercept DEMAND_SET self-target"))

    old_apply_pledge = r"""            elif act_type == "DEMAND_PLEDGE":
                from_p = act.get("from", speaker)
                target_p = act.get("target", "")
                field = act.get("field", "contribution")"""

    new_apply_pledge = r"""            elif act_type == "DEMAND_PLEDGE":
                from_p = act.get("from", speaker)
                target_p = act.get("target", "")
                if _is_owner(target_p, from_p):
                    continue
                field = act.get("field", "contribution")"""
    replacements.append((old_apply_pledge, new_apply_pledge, "6. _apply_actions skip self DEMAND_PLEDGE"))

    # 5. _format_action_badge
    old_badge = r"""            elif k == "DEMAND":
                d_type = a.get("demand_type")
                target = a.get("target", "всем")
                if d_type == "pledge":"""

    new_badge = r"""            elif k == "DEMAND":
                d_type = a.get("demand_type")
                target = a.get("target", "всем")
                if _is_owner(target, a.get("from", "")):
                    continue
                if d_type == "pledge":"""
    replacements.append((old_badge, new_badge, "7. _format_action_badge skip self demands"))

    # 6. _render_state
    old_render_state = r"""            else:
                lines.append(f"  • ⏱️ Значение: {val} мес. | {lev['title']} (В ведении: @{owner}{cost_note})")"""

    new_render_state = r"""            else:
                unit = " мес." if lev.get("type") == "int" else ""
                lines.append(f"  • ⏱️ Значение: {val}{unit} | {lev['title']} (В ведении: @{owner}{cost_note})")"""
    replacements.append((old_render_state, new_render_state, "8. _render_state unit formatting"))


# --- 7. _render_voter_checklist ---
old_checklist = r"""            f"3. Статус бюджета проекта: {'⚠️ ДЕФИЦИТ ' + _format_money(deficit) + ' (Для вступления Соглашения в силу дефицит должен быть закрыт!)' if deficit > 100_000.0 else '✅ ПОЛНОСТЬЮ ЗАКРЫТ (100% средств обеспечено)'}."""

new_checklist = r"""            f"3. Статус бюджета проекта: {'⚠️ ДЕФИЦИТ ' + _format_money(deficit) + ' (Принятие с дефицитом допустимо большинством голосов, покрытие ляжет на резервные фонды)' if deficit > 100_000.0 else '✅ ПОЛНОСТЬЮ ЗАКРЫТ (100% средств обеспечено)'}."""
replacements.append((old_checklist, new_checklist, "9. _render_voter_checklist deficit clause"))

# --- 8. _render_dialogue ---
old_dialogue_summary = r"""        passed = voting_summary.get("passed", False)
        votes_for = voting_summary.get("votes_for", 0)
        votes_against = voting_summary.get("votes_against", 0)
        votes_abstain = voting_summary.get("votes_abstain", 0)
        quorum = voting_summary.get("quorum", 5)
        majority_threshold = voting_summary.get("majority_threshold", (quorum // 2) + 1)
        deficit = voting_summary.get("deficit", max(0.0, total_cost - total_money))

        lines.append("\n---\n## 📜 ИТОГОВЫЙ ПРОТОКОЛ ОБЩЕСТВЕННЫХ СЛУШАНИЙ ГОРОДА N\n")
        if passed:
            lines.append(
                f"🎉 **СТАТУС: ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ!**\n"
                f"* **Итоги голосования:** За: **{votes_for}**, Против: **{votes_against}**, Воздержались: **{votes_abstain}** (Порог большинства: {majority_threshold} из {quorum})\n"
                f"* **Финансирование:** {_format_money(total_money)} из {_format_money(total_cost)} (Бюджет закрыт на 100%)\n"
            )
        else:
            reasons = []
            if votes_for < majority_threshold:
                reasons.append(f"не набрано большинство голосов (За: {votes_for}, необходимо: {majority_threshold})")
            if deficit > 100_000.0:
                reasons.append(f"не закрыт дефицит бюджета проекта ({_format_money(deficit)})")
            reason_str = "; ".join(reasons) if reasons else "несогласованность ключевых условий"
            lines.append(
                f"❌ **СТАТУС: ПРОТОКОЛ НЕ ПРИНЯТ!** (За: {votes_for}, Против: {votes_against}, Воздержались: {votes_abstain}).\n"
                f"* **Причина отклонения:** {reason_str}.\n"
            )

        lines.append("### 1. Согласованный финансовый план:")
        lines.append("| Участник | Согласованный взнос | Доля в проекте |")
        lines.append("| :--- | :--- | :--- |")

        total_share = 0
        for name, st in state["players"].items():
            c = st.get("contribution")
            s = st.get("share")
            c_str = _format_money(c)
            s_str = f"{s:g}%" if s is not None else "0%"
            if s:
                total_share += s
            lines.append(f"| **{name}** | {c_str} | {s_str} |")

        lines.append(f"\n* **Всего утверждено финансирования:** {_format_money(total_money)} (из требуемых {_format_money(total_cost)})")
        lines.append(f"* **Суммарная распределенная доля:** {total_share:g}%\n")

        lines.append("### 2. Утвержденные условия и обязательства сторон:")
        current_levers = state.get("levers", {})
        agreed_count = 0
        for lev in levers_config:
            lid = lev["id"]
            val = current_levers.get(lid, lev["default"])
            if lev["type"] == "bool":
                if val:
                    agreed_count += 1
                    desc_str = f" ({lev.get('description')})." if lev.get('description') else '.'
                    lines.append(f"* ✅ **{lev['title']}:** Подтверждено стороной @{lev['owner']}{desc_str}")
                else:
                    lines.append(f"* ❌ **{lev['title']}:** Не согласовано сторонами.")
            else:
                agreed_count += 1
                lines.append(f"* ⏱️ **{lev['title']}:** Утверждено на уровне: **{val}** (зафиксировано стороной @{lev['owner']}).")"""

new_dialogue_summary = r"""        passed = voting_summary.get("passed", False)
        votes_for = voting_summary.get("votes_for", 0)
        votes_against = voting_summary.get("votes_against", 0)
        votes_abstain = voting_summary.get("votes_abstain", 0)
        quorum = voting_summary.get("quorum", 5)
        majority_threshold = voting_summary.get("majority_threshold", (quorum // 2) + 1)
        deficit = voting_summary.get("deficit", max(0.0, total_cost - total_money))

        lines.append("\n---\n## 📜 ИТОГОВЫЙ ПРОТОКОЛ ОБЩЕСТВЕННЫХ СЛУШАНИЙ ГОРОДА N\n")
        if passed:
            lines.append(
                f"🎉 **СТАТУС: ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ!**\n"
                f"* **Итоги голосования:** За: **{votes_for}**, Против: **{votes_against}**, Воздержались: **{votes_abstain}** (Порог большинства: {majority_threshold} из {quorum})\n"
                f"* **Финансирование:** {_format_money(total_money)} из {_format_money(total_cost)} ({((total_money/total_cost)*100) if total_cost else 0:.1f}% от целевой суммы)\n"
            )
            if deficit > 100_000.0:
                lines.append(
                    f"> ⚠️ **Примечание по финансированию:** Проект принят при наличии остаточного дефицита в размере {_format_money(deficit)}. "
                    f"Недостающие средства подлежат согласованию отдельным протоколом из резервных фондов или дополнительных источников.\n"
                )
        else:
            reasons = []
            if votes_for < majority_threshold:
                reasons.append(f"не набрано большинство голосов (За: {votes_for}, необходимо: {majority_threshold})")
            if deficit > 100_000.0:
                reasons.append(f"не закрыт дефицит бюджета проекта ({_format_money(deficit)})")
            reason_str = "; ".join(reasons) if reasons else "несогласованность ключевых условий"
            lines.append(
                f"❌ **СТАТУС: ПРОТОКОЛ НЕ ПРИНЯТ!** (За: {votes_for}, Против: {votes_against}, Воздержались: {votes_abstain}).\n"
                f"* **Причина отклонения:** {reason_str}.\n"
            )

        lines.append("### 1. Согласованный финансовый план:")
        lines.append("| Участник | Согласованный взнос | Доля в проекте |")
        lines.append("| :--- | :--- | :--- |")

        total_share = 0
        for name, st in state["players"].items():
            c = st.get("contribution")
            s = st.get("share")
            c_val = c if c is not None else 0.0
            c_str = _format_money(c_val)
            s_str = f"{s:g}%" if s is not None and s > 0 else "0%"
            if s:
                total_share += s
            lines.append(f"| **{name}** | {c_str} | {s_str} |")

        lines.append(f"\n* **Всего утверждено финансирования:** {_format_money(total_money)} (из требуемых {_format_money(total_cost)})")
        lines.append(f"* **Суммарная распределенная доля:** {total_share:g}%\n")

        lines.append("### 2. Утвержденные условия и обязательства сторон:")
        current_levers = state.get("levers", {})
        agreed_count = 0
        for lev in levers_config:
            lid = lev["id"]
            val = current_levers.get(lid, lev["default"])
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
                lines.append(f"* {icon} **{title}:** Утверждено на уровне: **{val}{unit}** (зафиксировано стороной @{owner}).")"""
replacements.append((old_dialogue_summary, new_dialogue_summary, "10. _render_dialogue majority vote + formatting"))

# --- 9. _build_prompt_phase2 ---
old_build_phase2 = r"""        rules_block = f"\n\nРЕГЛАМЕНТ ЗАСЕДАНИЯ:\n{rules_prompt.strip()}" if rules_prompt else ""
        return (
            f"=== 💬 {banner} ===\n\n"
            f"ТВОЁ ИМЯ В ИГРЕ: {speaker}\n"
            f"СПИСОК ВСЕХ УЧАСТНИКОВ: {', '.join(state['players'].keys())}\n\n"
            f"ТВОЯ РОЛЬ И СТРАТЕГИЯ:\n{player_prompt}\n\n"
            f"ТВОЙ ЭКОНОМИЧЕСКИЙ ПРОФИЛЬ:\n{TwoLLMContractGame._render_profile(profile, state, speaker)}\n\n"
            f"{TwoLLMContractGame._render_player_levers(speaker, levers_config, state.get('levers', {}))}\n\n"
            f"{TwoLLMContractGame._render_state(state, levers_config, total_cost)}\n\n"
            f"{budget_guidance}"
            f"{demands_alert}\n"
            f"ИСТОРИЯ ПЕРЕГОВОРОВ:\n"
            f"{TwoLLMContractGame._render_history(history)}"
            f"{rules_block}\n\n"
            f"ИНСТРУКЦИЯ К ЭТАПУ 2:\n"
            f"Веди торг, отстаивай интересы своей стороны, предлагай компромиссы или выдвигай требования и ультиматумы.\n"
            f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\n"
            f"Ты можешь говорить в свободной форме или использовать удобные пометки:\n"
            f"- Свой взнос: прямо в речи ('увеличиваем наш вклад до 120 млн') или 'SET me Взнос <число_в_млн>'.\n"
            f"- Свои рычаги: прямо в речи ('согласны предоставить налоговые льготы') или 'SET <параметр> <значение>'.\n"
            f"- Требования и ультиматумы: четко назови сторону и условие (например: 'Требуем от холдинга 130 млн и 70% мощности').\n"
            f"- Обращение к партнеру: обратись по имени в речи или укажи 'TO @Имя_Игрока'."
        )"""

new_build_phase2 = r"""        # Анти-повторение: проверяем, говорил ли игрок ранее
        prev_speeches = [
            h.get("action", {}).get("message", "")
            for h in history
            if h.get("speaker") == speaker and h.get("action", {}).get("message")
        ]
        repetition_warning = ""
        if prev_speeches:
            repetition_warning = (
                "\n⚡ ПРАВИЛА ВЕДЕНИЯ ПЕРЕГОВОРОВ И АНТИ-ПОВТОРЕНИЕ:\n"
                "- Категорически запрещено повторять свою вступительную речь или прошлые реплики слово в слово!\n"
                "- Развивай диалог: отвечай на реплику собеседника, торгуйся, двигайся навстречу или жестко возражай.\n"
                "- Не вставляй служебные теги в текст речи (не пиши '[SET me ...]' внутри предложений, говори живо).\n"
            )

        rules_block = f"\n\nРЕГЛАМЕНТ ЗАСЕДАНИЯ:\n{rules_prompt.strip()}" if rules_prompt else ""
        return (
            f"=== 💬 {banner} ===\n\n"
            f"ТВОЁ ИМЯ В ИГРЕ: {speaker}\n"
            f"СПИСОК ВСЕХ УЧАСТНИКОВ: {', '.join(state['players'].keys())}\n\n"
            f"ТВОЯ РОЛЬ И СТРАТЕГИЯ:\n{player_prompt}\n\n"
            f"ТВОЙ ЭКОНОМИЧЕСКИЙ ПРОФИЛЬ:\n{TwoLLMContractGame._render_profile(profile, state, speaker)}\n\n"
            f"{TwoLLMContractGame._render_player_levers(speaker, levers_config, state.get('levers', {}))}\n\n"
            f"{TwoLLMContractGame._render_state(state, levers_config, total_cost)}\n\n"
            f"{budget_guidance}"
            f"{demands_alert}"
            f"{repetition_warning}\n"
            f"ИСТОРИЯ ПЕРЕГОВОРОВ:\n"
            f"{TwoLLMContractGame._render_history(history)}"
            f"{rules_block}\n\n"
            f"ИНСТРУКЦИЯ К ЭТАПУ 2:\n"
            f"Веди торг, отстаивай интересы своей стороны, предлагай компромиссы или выдвигай требования и ультиматумы.\n"
            f"Секретарь слушаний зафиксирует все твои слова, взносы и встречные условия в протоколе.\n"
            f"Ты можешь говорить в свободной форме или использовать удобные пометки:\n"
            f"- Свой взнос: прямо в речи ('увеличиваем наш вклад до 120 млн') или 'SET me Взнос <число_в_млн>'.\n"
            f"- Свои рычаги: прямо в речи ('согласны предоставить налоговые льготы') или 'SET <параметр> <значение>'.\n"
            f"- Требования и ультиматумы: четко назови сторону и условие (например: 'Требуем от холдинга 130 млн и 70% мощности').\n"
            f"- Обращение к партнеру: обратись по имени в речи или укажи 'TO @Имя_Игрока'."
        )"""
replacements.append((old_build_phase2, new_build_phase2, "11. _build_prompt_phase2 anti-repetition guidance"))

# --- 10. _run_phase2: rotation, starvation prevention, 1-turn cooldown, pair cap ---
old_run_phase2_rotation = r"""        pending_speaker_idx = None
        dialogue_pairs = []
        turns_silent = {p["name"]: 0 for p in players}

        for b_turn in range(max_turns):
            turn_counter += 1

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
                    if recent_count >= 4:
                        pending_speaker_idx = None

            if pending_speaker_idx is not None:
                current_idx = pending_speaker_idx
                pending_speaker_idx = None
            else:
                silent_longest = max(names, key=lambda p_name: (turns_silent[p_name], -name_to_idx[p_name]))
                current_idx = name_to_idx[silent_longest]

            current = players[current_idx]
            speaker = current["name"]

            for p_name in names:
                turns_silent[p_name] = 0 if p_name == speaker else turns_silent[p_name] + 1"""

new_run_phase2_rotation = r"""        pending_speaker_idx = None
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
            if starved_player is not None and (pending_speaker_idx is None or turns_silent.get(players[pending_speaker_idx]["name"], 0) < 3):
                current_idx = name_to_idx[starved_player]
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
                turns_silent[p_name] = 0 if p_name == speaker else turns_silent[p_name] + 1"""
replacements.append((old_run_phase2_rotation, new_run_phase2_rotation, "12. _run_phase2 starvation protection & rotation"))

old_phase2_system = r"""            system_message = (
                "Ты участник общественных слушаний по экологическому кризису в городе N. "
                "ИДЕТ ЭТАП 2: ПЕРЕГОВОРЫ И СОГЛАСОВАНИЕ УСЛОВИЙ. "
                f"{deficit_clause} "
                "Веди торг от лица своей роли: заявляй или изменяй свой взнос, переключай доступные условия, "
                "выдвигай встречные требования и ультиматумы к другим участникам. "
                "Обращайся к конкретным сторонам через @Имя или прямо в речи. "
                "Секретарь слушаний зафиксирует все изменения взносов, параметров и требований в протоколе."
            )"""

new_phase2_system = r"""            system_message = (
                "Ты участник общественных слушаний по экологическому кризису в городе N. "
                "ИДЕТ ЭТАП 2: ПЕРЕГОВОРЫ И СОГЛАСОВАНИЕ УСЛОВИЙ. "
                f"{deficit_clause} "
                "Веди живой диалог от лица своей роли: заявляй или изменяй свой взнос, переключай доступные условия, "
                "выдвигай встречные требования и ультиматумы к другим участникам. "
                "НЕ повторяй свои вступительные речи слово в слово — отвечай на реплики оппонентов и развивай компромисс! "
                "Обращайся к конкретным сторонам через @Имя или прямо в речи. "
                "Секретарь слушаний зафиксирует все изменения взносов, параметров и требований в протоколе."
            )"""
replacements.append((old_phase2_system, new_phase2_system, "13. _run_phase2 system_message anti-repetition"))

# --- 11. _run_phase3 ---
old_phase3_system_clause = r"""                f"Соглашение вступает в силу ТОЛЬКО при поддержке большинства (не менее {majority_threshold} из {num_players} голосов) и отсутствии дефицита бюджета."
            )"""

new_phase3_system_clause = r"""                f"Соглашение принимается большинством голосов (не менее {majority_threshold} из {num_players}). Дефицит бюджета (при наличии) фиксируется в протоколе для покрытия из резервных фондов."
            )"""
replacements.append((old_phase3_system_clause, new_phase3_system_clause, "14. _run_phase3 system_message majority threshold"))

old_phase3_vote_comment = r"""            vote_decision, vote_comment = self._parse_vote(command)
            vote_comment = _clean_speech_message(vote_comment)

            action = {"""

new_phase3_vote_comment = r"""            vote_decision, vote_comment = self._parse_vote(command)
            vote_comment = _clean_speech_message(vote_comment)
            if not vote_comment or vote_comment == "Подтверждаю свою позицию и ранее заявленные условия участия.":
                if vote_decision == "ПРОТИВ":
                    vote_comment = "Ключевые требования и красные линии стороны не были удовлетворены в итоговом соглашении."
                elif vote_decision == "ЗА":
                    vote_comment = "Компромиссные условия соглашения приемлемы для скорейшей реализации проекта."
                else:
                    vote_comment = "Поддерживаю необходимость проекта, но сохраняю замечания к отдельным обязательствам."

            action = {"""
replacements.append((old_phase3_vote_comment, new_phase3_vote_comment, "15. _run_phase3 vote comment contextual fallbacks"))

old_phase3_passed = r"""        votes_for = sum(1 for v in voting_records if v["vote"] == "ЗА")
        votes_against = sum(1 for v in voting_records if v["vote"] == "ПРОТИВ")
        votes_abstain = sum(1 for v in voting_records if v["vote"] == "ВОЗДЕРЖАЛСЯ")
        # Соглашение утверждено ТОЛЬКО при поддержке большинства и отсутствии дефицита бюджета
        budget_closed = (deficit <= 100_000.0)
        protocol_passed = (votes_for >= majority_threshold) and budget_closed"""

new_phase3_passed = r"""        votes_for = sum(1 for v in voting_records if v["vote"] == "ЗА")
        votes_against = sum(1 for v in voting_records if v["vote"] == "ПРОТИВ")
        votes_abstain = sum(1 for v in voting_records if v["vote"] == "ВОЗДЕРЖАЛСЯ")
        # Протокол принимается демократическим большинством голосов
        budget_closed = (deficit <= 100_000.0)
        protocol_passed = (votes_for >= majority_threshold)"""
replacements.append((old_phase3_passed, new_phase3_passed, "16. _run_phase3 democratic majority vote logic"))

modified = content
for old_s, new_s, desc in replacements:
    occurrences = modified.count(old_s)
    if occurrences == 0:
        print(f"ERROR: Could not find block for: {desc}")
        sys.exit(1)
    elif occurrences > 1:
        print(f"ERROR: Multiple occurrences ({occurrences}) found for: {desc}")
        sys.exit(1)
    else:
        modified = modified.replace(old_s, new_s)
        print(f"OK: Replaced {desc}")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(modified)

print("\nAll replacements successfully written to dialog.py!")

# Verify syntax
py_compile.compile(file_path, doraise=True)
print("dialog.py compiled cleanly with py_compile!")
