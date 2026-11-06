"""Локальное сохранение прогресса игры после каждого хода и при прерывании."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class RunStore:
    def __init__(self):
        self.run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_") + uuid4().hex[:8]
        root = Path(os.environ.get("NEGOTIATION_GAMES_DIR", "games")).expanduser().resolve()
        self.directory = root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.files = {"result.json": str(self.directory / "result.json"),
                      "result.md": str(self.directory / "result.md")}

    def save(self, payload: dict) -> None:
        payload["run_id"] = self.run_id
        payload["files"] = dict(self.files)
        contents = {"result.json": json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    "result.md": payload["report"]}
        for name, content in contents.items():
            target = self.directory / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(content + "\n", encoding="utf-8")
            temporary.replace(target)
