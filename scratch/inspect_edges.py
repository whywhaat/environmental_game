# -*- coding: utf-8 -*-
import json

with open("langflow/Negotiation_MVP_5_players.json", "r", encoding="utf-8") as f:
    flow = json.load(f)

edges = flow.get("data", {}).get("edges", [])
for e in edges:
    source = e.get("source")
    target = e.get("target")
    source_handle = e.get("data", {}).get("sourceHandle", {}).get("name")
    target_handle = e.get("data", {}).get("targetHandle", {}).get("name")
    print(f"{source} [{source_handle}] -> {target} [{target_handle}]")
