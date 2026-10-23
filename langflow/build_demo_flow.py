"""Refresh the Langflow demo export from the installed component templates.

Run after dumping ``build_custom_component_list_from_path`` to a JSON file.
"""

import copy
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TYPES = ("GameSettings", "PlayerAgent", "OpenAICompatibleModel", "TwoLLMContractGame", "ReportController")


def main(templates_file: Path) -> None:
    output = ROOT / "langflow" / "Negotiation_MVP_5_players.json"
    old = (json.loads(output.read_text(encoding="utf-8")) if output.exists() else
           json.loads(subprocess.check_output(["git", "show", "112c342^:LF_Agent_RAG.json"], cwd=ROOT)))
    templates = json.loads(templates_file.read_text(encoding="utf-8"))
    demo = json.loads((ROOT / "langflow" / "demo_config.json").read_text(encoding="utf-8"))
    old_nodes = {node["data"]["type"]: node for node in old["data"]["nodes"]}
    player_sample = next(node for node in old["data"]["nodes"] if node["data"]["type"] == "PlayerAgent")

    def node(kind, ident, x, y, values=None):
        base = copy.deepcopy(old_nodes.get(kind, player_sample))
        base["id"] = ident
        base["position"] = {"x": x, "y": y}
        base["data"]["id"] = ident
        base["data"]["type"] = kind
        info = copy.deepcopy(templates[kind])
        info.update({"lf_version": base["data"]["node"].get("lf_version", "1.5.0"), "category": "custom_components"})
        base["data"]["node"] = info
        base["data"]["selected_output"] = info["outputs"][0]["name"]
        field_template = info["template"]
        field_template["_frontend_node_flow_id"] = {"value": ident, "name": "_frontend_node_flow_id", "type": "str"}
        for key, value in (values or {}).items():
            if key in field_template:
                field_template[key]["value"] = value
        return base

    nodes = [node("GameSettings", "GameSettings-config", 0, 0, {
        "water_cost": 50, "monitoring_cost": 20, "reconstruction_cost": 200, "max_rounds": 15})]
    for index, player in enumerate(demo["players"]):
        y = index * 430
        nodes.append(node("OpenAICompatibleModel", f"OpenAICompatibleModel-{index}", 0, y + 250))
        nodes.append(node("PlayerAgent", f"PlayerAgent-{index}", 430, y, {
            "player_id": player["id"], "player_name": player["name"],
            "budget_millions": player["budget"] // 1_000_000,
            "expected_own_share": player["expected_own_share"],
            "desired_responsible": json.dumps(player["desired_responsible"], ensure_ascii=False),
            "personal_context": player["personal_context"], "motivation": player["motivation"],
            "personal_history": json.dumps(player["personal_history"], ensure_ascii=False),
        }))
    nodes.extend([
        node("OpenAICompatibleModel", "OpenAICompatibleModel-converter", 430, 2250),
        node("TwoLLMContractGame", "TwoLLMContractGame-main", 950, 900),
        node("ReportController", "ReportController-main", 1400, 900),
        copy.deepcopy(old_nodes["ChatOutput"]),
    ])
    nodes[-1]["position"] = {"x": 1850, "y": 900}
    by_id = {item["id"]: item for item in nodes}

    def edge(source, output, target, field):
        source_node = by_id[source]
        target_node = by_id[target]
        if source_node["data"]["type"] == "ChatOutput":
            raise ValueError("ChatOutput cannot be an edge source")
        source_output = next(item for item in source_node["data"]["node"]["outputs"] if item["name"] == output)
        source_handle = {"dataType": source_node["data"]["type"], "id": source, "name": output,
                         "output_types": source_output["types"]}
        target_input = target_node["data"]["node"]["template"][field]
        target_handle = {"fieldName": field, "id": target, "inputTypes": target_input["input_types"],
                         "type": target_input["type"]}
        encode = lambda data: json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace('"', "œ")
        sh, th = encode(source_handle), encode(target_handle)
        return {"id": f"reactflow__edge-{source}{sh}-{target}{th}", "source": source, "target": target,
                "sourceHandle": sh, "targetHandle": th,
                "data": {"sourceHandle": source_handle, "targetHandle": target_handle},
                "animated": False, "className": "", "selected": False}

    orchestrator = "TwoLLMContractGame-main"
    edges = [edge("GameSettings-config", "settings", orchestrator, "settings")]
    for index in range(5):
        edges.extend([
            edge(f"OpenAICompatibleModel-{index}", "model_output", f"PlayerAgent-{index}", "model"),
            edge(f"PlayerAgent-{index}", "player", orchestrator, "players"),
        ])
    edges.extend([
        edge("OpenAICompatibleModel-converter", "model_output", orchestrator, "converter_model"),
        edge(orchestrator, "result", "ReportController-main", "game_result"),
        edge("ReportController-main", "result", nodes[-1]["id"], "input_value"),
    ])
    old["name"] = "Negotiation MVP - five players"
    old["description"] = "Пять участников, три компонента, единое Python-ядро переговоров."
    old["data"] = {"nodes": nodes, "edges": edges, "viewport": {"x": 0, "y": 0, "zoom": 0.6}}
    output.write_bytes((json.dumps(old, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(output)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
