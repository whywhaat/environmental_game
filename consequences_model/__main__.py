"""Прогноз по сохранённой игре: python -m consequences_model games/<дата>/result.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import DEFAULT_PROMISED_MONTHS
from .model import forecast


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m consequences_model",
                                     description="Модель последствий: город N через 10 лет по итогу игры.")
    parser.add_argument("result", type=Path, help="result.json игры из каталога games/")
    parser.add_argument("--config", type=Path, help="конфигурация игры или список игроков: из неё берутся бюджеты")
    parser.add_argument("--months", type=int, default=DEFAULT_PROMISED_MONTHS, help="обещанный срок ввода очистных")
    parser.add_argument("--json", action="store_true", help="вывести весь прогноз в JSON вместо отчёта")
    args = parser.parse_args(argv)

    game_result = json.loads(args.result.read_text(encoding="utf-8"))
    players = None
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        players = config["players"] if isinstance(config, dict) else config
    result = forecast(game_result, players, args.months)
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["report"])
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
