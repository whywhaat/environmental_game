import json
import tempfile
import unittest
from pathlib import Path

from negotiation_mvp.core import GameConfig, State, merge_round, parse_sets, score_state
from negotiation_mvp.engine import ConversionError
from negotiation_mvp.langflow_bridge import run_langflow_game


DEMO = json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8"))


class BridgeTests(unittest.TestCase):
    def test_exported_graph_has_all_model_connections(self):
        flow = json.loads((Path(__file__).resolve().parents[1] / "langflow" / "Negotiation_MVP_5_players.json").read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in flow["data"]["nodes"]}
        edges = flow["data"]["edges"]
        self.assertEqual(sum(node["data"]["type"] == "PlayerAgent" for node in nodes.values()), 5)
        self.assertEqual(sum(node["data"]["type"] == "OpenAICompatibleModel" for node in nodes.values()), 6)
        self.assertEqual(len(edges), 14)
        for node in nodes.values():
            if node["data"]["type"] != "ChatOutput":
                self.assertIn(node["data"]["selected_output"], [item["name"] for item in node["data"]["node"]["outputs"]])
        for edge in edges:
            self.assertIn(edge["source"], nodes)
            self.assertIn(edge["target"], nodes)
            self.assertIn(edge["data"]["targetHandle"]["fieldName"], nodes[edge["target"]]["data"]["node"]["template"])

    def test_five_player_config_and_zero_budget_score(self):
        config = GameConfig.from_dict(DEMO)
        self.assertEqual(config.total_cost, 270_000_000)
        self.assertEqual(len(config.players), 5)
        state = merge_round(State(), parse_sets("SET participants.city_residents.contribution = 27000000", "city_residents", config), config)
        self.assertAlmostEqual(score_state(state, config)["city_residents"], -0.1)

    def test_routed_models_and_files(self):
        calls = {player["id"]: 0 for player in DEMO["players"]}

        def player_reply(player_id):
            def invoke(messages):
                calls[player_id] += 1
                return "VOTE ПРОТИВ" if any("Голосование по договору открыто" in item["content"] for item in messages) else "Без изменений."
            return invoke

        invoke = {player_id: player_reply(player_id) for player_id in calls}
        invoke["__converter__"] = lambda messages: "NO_CHANGES"
        with tempfile.TemporaryDirectory() as temp:
            output = run_langflow_game({**DEMO, "max_rounds": 1}, DEMO["players"], invoke, Path(temp))
            self.assertEqual(len(output["result"]["rounds"]), 1)
            self.assertEqual(set(output["result"]["votes"]), set(calls))
            self.assertTrue(all(value == 2 for value in calls.values()))
            self.assertTrue(all(Path(path).exists() for path in output["files"].values()))
            self.assertIn("Сообщения", Path(output["files"]["result_with_prompts.md"]).read_text(encoding="utf-8"))

    def test_conversion_error_has_no_final_json(self):
        invoke = {player["id"]: lambda messages: "Предлагаю 10 млн ₽." for player in DEMO["players"]}
        invoke["__converter__"] = lambda messages: 'SET participants.city_administration.contribution = "invalid"'
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ConversionError):
                run_langflow_game({**DEMO, "max_rounds": 1}, DEMO["players"], invoke, Path(temp))
            files = list(Path(temp).rglob("*"))
            self.assertFalse(any(path.name == "result.json" for path in files))
            self.assertEqual(len([path for path in files if path.suffix == ".md"]), 2)


if __name__ == "__main__":
    unittest.main()
