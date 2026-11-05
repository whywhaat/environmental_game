"""Обновить код NegotiationGame в базовом экспорте Langflow.

Запуск из корня репозитория: python langflow/update_negotiation_flow.py
"""

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPONENT = ROOT / "langflow/custom_components/my_components/dialog.py"
FLOW = ROOT / "langflow/Negotiation_MVP_5_players.json"


def main() -> None:
    code = COMPONENT.read_text(encoding="utf-8")
    ast.parse(code, filename=str(COMPONENT))
    flow = json.loads(FLOW.read_text(encoding="utf-8"))
    nodes = [node for node in flow["data"]["nodes"]
             if node["data"]["type"] == "NegotiationGame"]
    if len(nodes) != 1:
        raise ValueError(f"{FLOW}: ожидался ровно один узел NegotiationGame")
    nodes[0]["data"]["node"]["template"]["code"]["value"] = code
    updated = json.dumps(flow, ensure_ascii=False, indent=2) + "\n"
    if FLOW.read_text(encoding="utf-8") != updated:
        FLOW.write_text(updated, encoding="utf-8")
    print(f"Обновлён код NegotiationGame в {FLOW.name}")


if __name__ == "__main__":
    main()
