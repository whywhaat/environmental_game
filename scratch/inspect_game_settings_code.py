# -*- coding: utf-8 -*-
import json

with open("langflow/Negotiation_MVP_5_players.json", "r", encoding="utf-8") as f:
    flow = json.load(f)

for node in flow.get("data", {}).get("nodes", []):
    if node.get("id") == "GameSettings-config":
        template = node.get("data", {}).get("node", {}).get("template", {})
        code = template.get("code", {}).get("value", "")
        print("GameSettings code:\n", code)
