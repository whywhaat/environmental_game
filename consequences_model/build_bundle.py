"""Пакет подключения: только наш расчёт, примеры, компонент, стенд, изображения и инструкция."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from .build_component import COMPONENT_PATH, render_component
from .build_demo import DEMO_FLOW_PATH, SELECTOR, render_selector
from .build_flow import NODE
from .catalog import OUTCOMES

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "dist/consequences-model.zip"
RUNTIME_MODULES = ("__init__", "__main__", "catalog", "protocol", "state", "report", "explain", "model")
README = """# Модель последствий — пакет подключения

Начните с [инструкции](docs/consequences-integration.md).

Из корня распакованного архива, Python 3.11+:

```bash
python -m consequences_model examples/consequences/open_river.json --json
```

Для игрока используйте поле `herald_text`.
В `examples/consequences/` — десять синтетических примеров исходов;
в [examples/integration/](examples/integration/README.md) — компактные итоги двух реальных партий.
Стенд: `langflow/Consequences_demo.json`; PNG загружаются в его поля отдельно.
"""


def build_bundle(output: Path = ARCHIVE_PATH) -> Path:
    """Собрать переносимый архив по разрешённым путям, без локального окружения и ядра игры."""
    component = COMPONENT_PATH.read_text(encoding="utf-8")
    if component != render_component():
        raise ValueError("Компонент устарел: выполните python -m consequences_model.build_component")
    demo = json.loads(DEMO_FLOW_PATH.read_text(encoding="utf-8"))
    code = {node["id"]: node["data"]["node"]["template"]["code"]["value"] for node in demo["data"]["nodes"]}
    if code.get(NODE) != component or code.get(SELECTOR) != render_selector():
        raise ValueError("Стенд устарел: пересоберите build_flow и build_demo перед упаковкой")
    paths = [Path("consequences_model") / f"{module}.py" for module in RUNTIME_MODULES]
    paths += [Path(f"examples/consequences/{key}.json") for key in OUTCOMES]
    paths += [Path(f"assets/consequences/{key}.png") for key in OUTCOMES
              if (ROOT / f"assets/consequences/{key}.png").is_file()]
    paths += [Path("examples/consequences/manifest.json"), Path("docs/consequences-integration.md"),
              Path("examples/integration/README.md"),
              Path("examples/integration/vlad_10a5c68_accepted.json"),
              Path("examples/integration/vlad_10a5c68_rejected.json"),
              Path("langflow/Consequences_demo.json"),
              Path("langflow/custom_components/my_components/ConsequencesModel.py")]
    payload = {path.as_posix(): (ROOT / path).read_bytes() for path in paths}
    payload["README.md"] = README.encode("utf-8")
    manifest = {"format_version": 1, "python": ">=3.11", "langflow_tested": "1.12.5",
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in sorted(payload.items())}}
    payload["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for name, content in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            entry.create_system = 3
            bundle.writestr(entry, content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Собрать пакет подключения нашей модели последствий.")
    parser.add_argument("--output", type=Path, default=ARCHIVE_PATH, help="путь к ZIP, по умолчанию dist/consequences-model.zip")
    args = parser.parse_args()
    print(f"Собрано: {build_bundle(args.output)}")


if __name__ == "__main__":
    main()
