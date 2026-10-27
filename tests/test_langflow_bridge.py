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
        self.assertEqual(sum(node["data"]["type"] == "SharedProjectRAG" for node in nodes.values()), 1)
        self.assertEqual(sum(node["data"]["type"] == "LanguageModelComponent" for node in nodes.values()), 1)
        self.assertEqual(len(edges), 10)
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
        self.assertAlmostEqual(score_state(state, config)["city_residents"], 40.9)

    def test_shared_funding_score_with_no_player_budget(self):
        config = GameConfig.from_dict({**DEMO, "players": [{**player, "budget": 0} for player in DEMO["players"]]})
        self.assertTrue(all(score == 40 for score in score_state(State(), config).values()))

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

    def test_rag_retrieval_and_integration(self):
        from negotiation_mvp.rag import RAGRetriever
        retriever = RAGRetriever()
        ctx_admin = retriever.get_context("city_administration", 1, "очистные сооружения")
        self.assertIn("СПРАВКА ИЗ БАЗЫ ПРОЕКТА (RAG)", ctx_admin)
        self.assertIn("[", ctx_admin)
        self.assertGreater(len(retriever.audit), 0)

        received_messages = []
        def capturing_invoke(messages):
            received_messages.append(messages)
            return "Без изменений."

        invoke = {player["id"]: capturing_invoke for player in DEMO["players"]}
        invoke["__converter__"] = lambda messages: "NO_CHANGES"
        with tempfile.TemporaryDirectory() as temp:
            output = run_langflow_game({**DEMO, "max_rounds": 1}, DEMO["players"], invoke, Path(temp), rag_config={"top_k": 2})
            self.assertTrue(len(output["result"]["rag_audit"]) > 0)
            has_rag = any(
                any("СПРАВКА ИЗ БАЗЫ ПРОЕКТА (RAG)" in m["content"] for m in call)
                for call in received_messages
            )
            self.assertTrue(has_rag)

    def test_simplified_player_config_with_defaults(self):
        simplified = {
            "max_rounds": 1,
            "components": DEMO["components"],
            "players": [
                {"id": "admin", "name": "Администрация", "budget": 30_000_000, "prompt": "Добивайся выгодного соглашения."},
                {"id": "holding", "name": "Холдинг", "budget": 180_000_000, "prompt": "Ограничь свои расходы."},
            ]
        }
        config = GameConfig.from_dict(simplified)
        self.assertEqual(len(config.players), 2)
        self.assertEqual(config.players[0].expected_own_share, 0.0)
        self.assertEqual(set(config.players[0].desired_responsible), config.component_ids)
        self.assertEqual(config.players[0].personal_history, ())
        self.assertIn("Добивайся выгодного соглашения", config.players[0].personal_context)

        invokers = {"admin": lambda m: "Без изменений.", "holding": lambda m: "Без изменений.", "__converter__": lambda m: "NO_CHANGES"}
        with tempfile.TemporaryDirectory() as temp:
            output = run_langflow_game(simplified, simplified["players"], invokers, Path(temp))
            self.assertEqual(len(output["result"]["rounds"]), 1)


if __name__ == "__main__":
    unittest.main()
