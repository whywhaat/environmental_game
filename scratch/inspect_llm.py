# -*- coding: utf-8 -*-
import json

with open("langflow/Negotiation_MVP_5_players.json", "r", encoding="utf-8") as f:
    flow = json.load(f)

for node in flow.get("data", {}).get("nodes", []):
    nid = node.get("id")
    template = node.get("data", {}).get("node", {}).get("template", {})
    if "llm" in template:
        llm_val = template["llm"].get("value")
        print(f"Node {nid}: template['llm'] = {type(llm_val)}: {llm_val}")
    if "model_name" in template:
        print(f"Node {nid}: template['model_name'] = {template['model_name'].get('value')}")
    if nid == "LanguageModelComponent-converter":
        for k, v in template.items():
            if isinstance(v, dict) and "value" in v:
                print(f"  Converter {k}: {v.get('value')}")
