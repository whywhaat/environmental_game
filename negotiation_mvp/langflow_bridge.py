"""Small boundary between Langflow model objects and the game engine."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .core import GameConfig
from .engine import run_game
from .llm import RecordingChat
from .report import append_prompts, render_interrupted_report, render_prompt_report, render_report


class RoutedChat:
    def __init__(self, invoke: dict[str, Callable[[list[dict[str, str]]], str]]):
        self.invoke = invoke

    def complete(self, model: str, messages: list[dict[str, str]]) -> str:
        return self.invoke[model](messages)


def run_langflow_game(
    settings: dict[str, Any],
    players: list[dict[str, Any]],
    invoke: dict[str, Callable[[list[dict[str, str]]], str]],
    games_dir: Path,
    converter_model_name: str | None = None,
) -> dict[str, Any]:
    """Validate the graph inputs, run the same engine as CLI, and persist artifacts."""
    raw_config = {"components": settings["components"], "max_rounds": settings["max_rounds"],
                  "players": [{key: value for key, value in player.items() if key != "llm"} for player in players]}
    config = GameConfig.from_dict(raw_config)
    required_models = config.player_ids | {"__converter__"}
    if set(invoke) != required_models:
        raise ValueError(f"Model connections must match players and converter: {sorted(required_models)}")
    run_dir = games_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    chat = RecordingChat(RoutedChat(invoke))
    try:
        result = run_game(config, chat, {player.id: player.id for player in config.players}, "__converter__")
        result["models"] = {
            "players": {player["id"]: str(getattr(player.get("llm"), "model_name", player["id"])) for player in players},
            "converter": converter_model_name or "__converter__",
        }
    except Exception as exc:
        interrupted = render_interrupted_report(config, exc)
        (run_dir / "result.md").write_text(interrupted, encoding="utf-8")
        (run_dir / "result_with_prompts.md").write_text(append_prompts(interrupted, chat.calls), encoding="utf-8")
        raise
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "result.md").write_text(render_report(result, config), encoding="utf-8")
    (run_dir / "result_with_prompts.md").write_text(render_prompt_report(result, config, chat.calls), encoding="utf-8")
    return {"result": result, "files": {name: str(run_dir / name) for name in
            ("result.json", "result.md", "result_with_prompts.md")}, "report": render_report(result, config)}
