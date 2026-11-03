"""Текстовое представление игры: отчёт, протокол, история и сводки для промптов.

Функции читают переданные данные и возвращают строки; не вызывают модели и не
изменяют состояние. Модуль не зависит от Langflow или LangChain.
"""

from .rules import is_owner


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
        lines.append(f"  - ID для own_levers: {lev['id']}; тип: {lev['type']}")
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
        "1. Юридическую силу имеют ТОЛЬКО взносы из таблицы выше. Речь не меняет обязательства: новый взнос задаётся полем own_contribution_millions.",
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
            vote = act.get("vote") or "ГОЛОС НЕ ПОЛУЧЕН"
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
        lines.append(
            f"🎉 **СТАТУС: ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ!**\n"
            f"* **Итоги голосования:** За: **{votes_for}**, Против: **{votes_against}**, Воздержались: **{votes_abstain}** (Порог большинства: {majority_threshold} из {quorum})\n"
            f"* **Финансирование:** {format_money(total_money)} из {format_money(total_cost)} ({((total_money/total_cost)*100) if total_cost else 0:.1f}% от целевой суммы)\n"
        )
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

