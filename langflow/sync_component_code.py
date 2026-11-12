"""Copy local Langflow component code into both exported flows.

Run from the repository root: python langflow/sync_component_code.py
Generated dialog.py and ConsequencesModel.py must be rebuilt first.
"""

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
COMPONENTS = HERE / "custom_components" / "my_components"
flow_path = (HERE / "Negotiation_MVP_5_players.json")


def component_codes():
    codes = {}
    sources = {}
    for path in COMPONENTS.glob("*.py"):
        if path.name == "__init__.py":
            continue
        code = path.read_text(encoding="utf-8")
        classes = {node.name for node in ast.parse(code).body if isinstance(node, ast.ClassDef)}
        # A component file may contain helper classes; the component name is
        # determined by matching a class to the node type in the flow.
        for name in classes:
            if name in codes:
                raise ValueError(f"Duplicate component class {name}: {path}")
            codes[name] = code
            sources[name] = path.name
    return codes, sources


def main():
    codes, sources = component_codes()
    pending = []
    used_types = set()
    path = flow_path
    original = path.read_bytes()
    newline = "\r\n" if b"\r\n" in original else "\n"
    flow = json.loads(original.decode("utf-8"))
    changed = 0
    for node in flow["data"]["nodes"]:
        template = node["data"]["node"].get("template", {})
        code_field = template.get("code")
        if not isinstance(code_field, dict):
            continue
        node_type = node["data"]["type"]
        if node_type == "ChatOutput":
            continue
        if node_type not in codes:
            raise ValueError(f"{path.name}: no local .py for component {node_type}")
        used_types.add(node_type)
        if code_field.get("value") != codes[node_type]:
            code_field["value"] = codes[node_type]
            changed += 1
    if changed:
        content = (json.dumps(flow, ensure_ascii=False, indent=2) + "\n").replace("\n", newline)
        pending.append((path, content.encode("utf-8")))
    print(f"{path.name}: {changed} nodes updated")
    for path, content in pending:
        path.write_bytes(content)
    unused_files = sorted({path.name for path in COMPONENTS.glob("*.py")
                           if path.name != "__init__.py" and path.name not in
                           {sources[name] for name in used_types}})
    if unused_files:
        print("No node in either flow for: " + ", ".join(unused_files))


if __name__ == "__main__":
    main()
