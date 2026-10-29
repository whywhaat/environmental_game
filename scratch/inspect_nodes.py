# -*- coding: utf-8 -*-
import json

with open("langflow/Negotiation_MVP_5_players.json", "r", encoding="utf-8") as f:
    flow = json.load(f)

nodes = flow.get("data", {}).get("nodes", [])
print(f"Total nodes: {len(nodes)}")
for n in nodes:
    data = n.get("data", {})
    node_type = data.get("type")
    node_name = data.get("node", {}).get("display_name", n.get("id"))
    print(f"- {n.get('id')}: {node_name} ({node_type})")
