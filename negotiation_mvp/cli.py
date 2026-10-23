"""Command-line runner for the standalone MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .core import GameConfig
from .engine import run_game
from .llm import OpenAICompatibleChat, RecordingChat
from .report import append_prompts, render_interrupted_report, render_prompt_report, render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run negotiation MVP with an OpenAI-compatible chat API")
    parser.add_argument("--config", required=True, type=Path, help="Game configuration JSON")
    parser.add_argument("--output", type=Path, help="JSON filename inside this run's directory (default: result.json)")
    parser.add_argument("--report", type=Path, help="Markdown filename inside this run's directory (default: beside JSON)")
    parser.add_argument("--prompt-report", type=Path, help="Filename for the report with all LLM messages and replies")
    parser.add_argument("--games-dir", type=Path, default=Path("games"), help="Directory for saved game runs (default: games)")
    args = parser.parse_args(argv)
    config = GameConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    player_model = os.getenv("PLAYER_MODEL")
    converter_model = os.getenv("CONVERTER_MODEL")
    if not player_model or not converter_model:
        parser.error("PLAYER_MODEL and CONVERTER_MODEL are required")
    run_dir = args.games_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    output_path = run_dir / (args.output.name if args.output else "result.json")
    report_path = run_dir / (args.report.name if args.report else output_path.with_suffix(".md").name)
    prompt_report_path = run_dir / (args.prompt_report.name if args.prompt_report else f"{report_path.stem}_with_prompts.md")
    if prompt_report_path == report_path:
        parser.error("--prompt-report must differ from --report")
    chat = RecordingChat(OpenAICompatibleChat())
    try:
        result = run_game(config, chat, player_model, converter_model)
    except Exception as exc:
        interrupted = render_interrupted_report(config, exc)
        report_path.write_text(interrupted, encoding="utf-8")
        prompt_report_path.write_text(append_prompts(interrupted, chat.calls), encoding="utf-8")
        raise
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    report_path.write_text(render_report(result, config), encoding="utf-8")
    prompt_report_path.write_text(render_prompt_report(result, config, chat.calls), encoding="utf-8")
    output_path.write_text(rendered + "\n", encoding="utf-8")
    if not args.output:
        sys.stdout.write(rendered + "\n")
    return 0
