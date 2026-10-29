# -*- coding: utf-8 -*-
import json

with open("langflow/Negotiation_MVP_5_players.json", "r", encoding="utf-8") as f:
    flow = json.load(f)

for node in flow.get("data", {}).get("nodes", []):
    if "GameSettings" in node.get("id", "") or node.get("data", {}).get("type") == "GameSettings":
        print(f"Found node: {node.get('id')}")
        template = node.get("data", {}).get("node", {}).get("template", {})
        for k, v in template.items():
            if isinstance(v, dict) and "value" in v:
                val = v.get("value")
                val_str = str(val)[:200] if val is not None else "None"
                print(f"  Field {k}: {val_str}")
                if k == "levers":
                    print("  FULL LEVERS:", json.dumps(val, indent=2, ensure_ascii=False))
