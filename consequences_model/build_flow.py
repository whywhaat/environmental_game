"""Собирает свой flow: граф переговоров + узел «Модель последствий».

Граф переговоров берётся из langflow/Negotiation_MVP_5_players.json без изменений: в нём ничего не
правится, только добавляются узел модели последствий, его ChatOutput и связи. Шаблон узла строит lfx
по собранному файлу компонента, поэтому сначала пересоберите компонент (build_component), затем flow:

    uv run --no-project --python 3.12 --with "lfx==1.12.5" python -m consequences_model.build_flow
"""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

from .build_component import COMPONENT_PATH

LANGFLOW = COMPONENT_PATH.parents[2]
SOURCE_FLOW_PATH = LANGFLOW / "Negotiation_MVP_5_players.json"
FLOW_PATH = LANGFLOW / "Negotiation_MVP_with_consequences.json"
ORCHESTRATOR = "TwoLLMContractGame-main"
NODE = "ConsequencesModel-main"
CHAT = "ChatOutput-consequences"
FLOW_NAME = "Negotiation MVP + модель последствий"
FLOW_DESCRIPTION = "Переговоры пяти участников и прогноз модели последствий: город N через 10 лет."


def component_template(code: str) -> dict[str, Any]:
    """Шаблон узла, как его строит сам Langflow; нужен установленный lfx."""
    from lfx.custom.custom_component.component import Component
    from lfx.custom.utils import build_custom_component_template

    template, _ = build_custom_component_template(Component(_code=code))
    return template


def build_flow(source: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    flow = copy.deepcopy(source)
    nodes = flow["data"]["nodes"]
    by_id = {node["id"]: node for node in nodes}
    by_type = {node["data"]["type"]: node for node in nodes}

    # Узел модели повторяет устройство узла отчёта Олега: тоже свой компонент, читающий итог игры.
    report = by_type["ReportController"]
    model = copy.deepcopy(report)
    x, y = report["position"]["x"], report["position"]["y"] + report["measured"]["height"] + 80
    info = copy.deepcopy(template)
    info.update({"lf_version": report["data"]["node"]["lf_version"], "category": "custom_components"})
    info["template"]["_frontend_node_flow_id"] = {"value": NODE, "name": "_frontend_node_flow_id", "type": "str"}
    model.update({"id": NODE, "position": {"x": x, "y": y}})
    model["data"].update({"id": NODE, "type": "ConsequencesModel", "node": info, "selected_output": "report",
                          "display_name": info["display_name"], "description": info["description"]})

    chat = copy.deepcopy(by_type["ChatOutput"])
    chat.update({"id": CHAT, "position": {"x": chat["position"]["x"], "y": y}})
    chat["data"]["id"] = CHAT

    nodes.extend([model, chat])
    by_id.update({NODE: model, CHAT: chat})

    def edge(source_id: str, output: str, target_id: str, field: str) -> dict[str, Any]:
        # Та же кодировка ручек, что в langflow/build_demo_flow.py.
        source_node, target_node = by_id[source_id], by_id[target_id]
        source_output = next(item for item in source_node["data"]["node"]["outputs"] if item["name"] == output)
        source_handle = {"dataType": source_node["data"]["type"], "id": source_id, "name": output,
                         "output_types": source_output["types"]}
        target_input = target_node["data"]["node"]["template"][field]
        target_handle = {"fieldName": field, "id": target_id, "inputTypes": target_input["input_types"],
                         "type": target_input["type"]}

        def encode(data: dict[str, Any]) -> str:
            return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace('"', "œ")

        sh, th = encode(source_handle), encode(target_handle)
        return {"id": f"reactflow__edge-{source_id}{sh}-{target_id}{th}", "source": source_id, "target": target_id,
                "sourceHandle": sh, "targetHandle": th,
                "data": {"sourceHandle": source_handle, "targetHandle": target_handle},
                "animated": False, "className": "", "selected": False}

    players = [node["id"] for node in nodes if node["data"]["type"] == "PlayerAgent"]
    flow["data"]["edges"].extend([
        edge(ORCHESTRATOR, "result", NODE, "game_result"),
        *(edge(player, "player", NODE, "players") for player in players),
        edge(NODE, "report", CHAT, "input_value"),
    ])
    # Свой id, чтобы импорт не перезаписал flow переговоров, и тот же при каждой пересборке.
    flow.update({"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source['id']}/consequences")),
                 "name": FLOW_NAME, "description": FLOW_DESCRIPTION})
    return flow


def write_flow(flow: dict[str, Any], path: Path = FLOW_PATH) -> None:
    path.write_bytes((json.dumps(flow, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def main() -> None:
    source = json.loads(SOURCE_FLOW_PATH.read_text(encoding="utf-8"))
    write_flow(build_flow(source, component_template(COMPONENT_PATH.read_text(encoding="utf-8"))))
    print(f"Собрано: {FLOW_PATH}")


if __name__ == "__main__":
    main()
