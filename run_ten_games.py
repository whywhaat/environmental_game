"""Run the same negotiation scenario once for each of ten player models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config.metrics import PLAYER_MODELS


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ten separate games with different player models")
    parser.add_argument("--config", type=Path, default=Path("example_config.json"))
    parser.add_argument("--games-dir", type=Path, default=Path("games"))
    parser.add_argument("--start-at", type=int, default=1, help="First game number to run (1-10)")
    parser.add_argument("--openrouter-key", help="OpenRouter API key (or set OPENROUTER_KEY in .env)")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_KEY")
    if not openrouter_key:
        parser.error("Provide --openrouter-key or OPENROUTER_KEY in .env")

    models = PLAYER_MODELS
    if len(models) != 10:
        parser.error(f"config.metrics.PLAYER_MODELS must contain exactly 10 model IDs; found {len(models)}")
    if not 1 <= args.start_at <= 10:
        parser.error("--start-at must be between 1 and 10")
    if any(not isinstance(model, str) or not model.strip() or model.startswith("model-provider/") for model in models):
        parser.error("Replace all placeholder model IDs in config/metrics.py")
    if not args.config.is_file():
        parser.error(f"Game config not found: {args.config}")

    batch_dir = args.games_dir / ("batch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
    batch_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for number, model in enumerate(models, 1):
        if number < args.start_at:
            continue
        game_dir = batch_dir / f"game_{number:02d}"
        environment = os.environ.copy()
        environment["PLAYER_MODEL"] = model
        environment["PLAYER_OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        environment["PLAYER_OPENAI_API_KEY"] = openrouter_key
        environment["PLAYER_OPENAI_SSL_VERIFY"] = "true"
        command = [sys.executable, "-m", "negotiation_mvp", "--config", str(args.config.resolve()),
                   "--games-dir", str(game_dir.resolve()), "--output", "result.json"]
        print(f"[{number}/10] {model}", flush=True)
        completed = subprocess.run(command, env=environment, text=True, capture_output=True)
        reports = list(game_dir.glob("*/result.md"))
        result_file = next(iter(game_dir.glob("*/result.json")), None)
        entry = {"number": number, "player_model": model, "status": "ok" if completed.returncode == 0 else "error",
                 "report": str(reports[0]) if reports else None,
                 "result": str(result_file) if result_file else None}
        if result_file:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            entry.update(accepted=data["accepted"], final_scores=data["final_scores"])
        if completed.returncode:
            entry["error"] = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else f"Exit code {completed.returncode}"
        results.append(entry)
        (batch_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  {entry['status']}: {entry.get('error') or entry['report']}", flush=True)
    print(f"Summary: {batch_dir / 'summary.json'}")
    return 1 if any(entry["status"] == "error" for entry in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
