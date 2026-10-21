"""Human-readable Markdown report from a completed game result."""

from __future__ import annotations

import re
from typing import Any

from .core import GameConfig


def _money(value: int) -> str:
    return f"{value / 1_000_000:g} млн ₽"


def _speech(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.strip().splitlines()) or "> —"


def _score(value: float) -> str:
    return f"{value:+g}".replace(".", ",")


def _field_name(path: str, config: GameConfig) -> str:
    names = {player.id: player.name for player in config.players}
    components = {component.id: component.name for component in config.components}
    parts = path.split(".")
    if parts[0] == "participants":
        return f"Взнос: {names.get(parts[1], parts[1])}"
    return f"Ответственный: {components.get(parts[1], parts[1])}"


def render_report(result: dict[str, Any], config: GameConfig) -> str:
    names = {player.id: player.name for player in config.players}
    lines = [
        "# Отчёт о переговорах",
        "",
        f"**Стоимость проекта:** {_money(config.total_cost)}.",
        "**Состав проекта:** " + "; ".join(f"{component.name} — {_money(component.cost)}" for component in config.components) + ".",
        "**Участники:** " + ", ".join(names.values()) + ".",
    ]
    for round_data in result["rounds"]:
        lines.extend(["", f"## Раунд {round_data['number']}"])
        for speech in round_data["speeches"]:
            lines.extend([
                "", f"### {names[speech['speaker']]}", "",
                _speech(speech["speech"]), "",
                "**Извлечённые SET:**", "", "```text",
                speech["sets"].strip() or "NO_CHANGES", "```",
            ])
        conflicts = round_data["conflicts"]
        lines.extend([
            "", "### Итог раунда", "",
            round_data["table"], "",
            f"**Согласованное финансирование:** {_money(round_data['funding_total'])} из {_money(config.total_cost)}.",
            "**Конфликты:** " + ("; ".join(_field_name(path, config) for path in conflicts) if conflicts else "нет") + ".",
            "**Score:** " + "; ".join(f"{names[player_id]} {_score(value)}" for player_id, value in round_data["scores"].items()) + ".",
        ])
    lines.extend(["", "## Голосование", ""])
    for vote in (turn for turn in result["turns"] if turn.get("phase") == "vote"):
        player_id = vote["speaker"]
        decision = "за" if result["votes"][player_id] else "против"
        lines.extend([f"### {names[player_id]} — {decision}", "", _speech(vote["speech"]), ""])
    final_round = result["rounds"][-1]
    lines.extend([
        "## Итог", "",
        f"**Решение:** {'договор принят' if result['accepted'] else 'договор не принят'}.",
        f"**Согласованное финансирование:** {_money(final_round['funding_total'])} из {_money(config.total_cost)}.",
        "**Конфликты:** " + ("; ".join(_field_name(path, config) for path in final_round["conflicts"]) if final_round["conflicts"] else "нет") + ".",
        "**Итоговый Score:** " + "; ".join(f"{names[player_id]} {_score(value)}" for player_id, value in result["final_scores"].items()) + ".",
        "", "### Итоговая таблица", "", result["final_table"], "",
    ])
    return "\n".join(lines)


def _code_block(content: str) -> str:
    longest_run = max((len(run) for run in re.findall(r"`+", content)), default=0)
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}text\n{content}\n{fence}"


def append_prompts(report: str, calls: list[dict[str, Any]]) -> str:
    lines = [report.rstrip(), "", "## Сообщения, переданные LLM", ""]
    for number, call in enumerate(calls, 1):
        recipient = "конвертер" if call["messages"] and call["messages"][0]["content"].startswith("Ты конвертер") else "игрок"
        lines.extend([f"### Вызов {number} — {recipient}, модель {call['model']}", ""])
        for message in call["messages"]:
            lines.extend([f"**{message['role']}**", "", _code_block(message["content"]), ""])
        lines.extend(["**Ответ модели**", "", _code_block(call.get("response", call.get("error", ""))), ""])
    return "\n".join(lines)


def render_prompt_report(result: dict[str, Any], config: GameConfig, calls: list[dict[str, Any]]) -> str:
    return append_prompts(render_report(result, config), calls)


def render_interrupted_report(config: GameConfig, error: Exception) -> str:
    return (
        "# Отчёт о переговорах\n\n"
        "**Статус:** запуск технически прерван; итогового решения нет.\n\n"
        f"**Стоимость проекта:** {_money(config.total_cost)}.\n\n"
        f"**Ошибка:** {type(error).__name__}: {error}\n"
    )
