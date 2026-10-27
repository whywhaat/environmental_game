"""Exercise the real parser/action methods without importing Langflow services."""

import ast
import copy
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def action_helpers(code):
    tree = ast.parse(code)
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_is_owner")
    game = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TwoLLMContractGame")
    methods = [node for node in game.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_parse_single_set", "_apply_actions"}]
    for method in methods:
        method.decorator_list = []
    module = ast.Module(body=[owner, *methods], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, "<langflow action methods>", "exec"), namespace)
    return namespace


class LangflowActionTests(unittest.TestCase):
    def test_role_aliases_apply_to_existing_player_in_source_and_export(self):
        graph = json.loads((ROOT / "langflow/Negotiation_MVP_5_players.json").read_text())
        node = next(node for node in graph["data"]["nodes"] if node["data"]["type"] == "TwoLLMContractGame")
        versions = {
            "source": (ROOT / "langflow/custom_components/my_components/dialog.py").read_text(),
            "export": node["data"]["node"]["template"]["code"]["value"],
        }
        for version, code in versions.items():
            helpers = action_helpers(code)
            for kind, value, cost in (("bool", "нет", 0), ("bool", "да", 20),
                                      ("int", "24", 0), ("str", "shared", 0)):
                with self.subTest(version=version, kind=kind, value=value):
                    state = {"players": {"Жители": {"spent_on_levers": 0}}, "levers": {}}
                    lever = {"id": "test", "title": "Условие", "type": kind,
                             "owner": "Инициативная группа жителей", "cost": cost, "default": 0}
                    action = helpers["_parse_single_set"](
                        f"SET test {value}", "Жители", state, [lever], 270_000_000)
                    self.assertEqual(action["type"], "LEVER_SET")
                    self.assertEqual(action["player"], "Жители")
                    helpers["_apply_actions"](state, {"actions": [action]}, None, 270_000_000)
                    self.assertEqual(set(state["players"]), {"Жители"})
                    self.assertEqual(state["players"]["Жители"]["spent_on_levers"], cost)
                    before = copy.deepcopy(state)
                    rejected = helpers["_parse_single_set"](
                        f"SET test {value}", "Промкомплекс", state, [lever], 270_000_000)
                    self.assertEqual(rejected["type"], "LEVER_REJECTED")
                    helpers["_apply_actions"](state, {"actions": [rejected]}, None, 270_000_000)
                    self.assertEqual(state, before)
