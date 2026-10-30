"""Прогноз по сохранённому итогу игры: python -m consequences_model game.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .model import forecast


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m consequences_model",
                                     description="Модель последствий: город N через 10 лет по итогу игры.")
    parser.add_argument("result", type=Path, help="итог игры: выход оркестратора (Data.data), сохранённый в JSON")
    parser.add_argument("--json", action="store_true", help="вывести весь прогноз в JSON вместо отчёта")
    args = parser.parse_args(argv)

    result = forecast(json.loads(args.result.read_text(encoding="utf-8")))
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["report"])
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
