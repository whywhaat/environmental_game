"""Проверочные итоги можно выбирать и рассчитывать независимо от игры."""

import json
import unittest

from consequences_model import forecast
from consequences_model.build_demo import DEMO_FLOW_PATH, build_demo, choose_example, read_examples, render_selector
from consequences_model.build_component import COMPONENT_PATH
from consequences_model.build_flow import FLOW_PATH


class ConsequencesDemoTests(unittest.TestCase):
    def test_a_chat_choice_supplies_the_corresponding_game_result(self):
        cases = read_examples()
        for query in ("1", "open_river", "  ОТКРЫТАЯ РЕКА  ", "", "   "):
            with self.subTest(query=query):
                result = forecast(choose_example(query, cases)["game_result"])
                self.assertEqual(result["outcome"]["id"], "open_river")
                self.assertEqual(result["paid"], 270_000_000)
                self.assertEqual(result["commissioning_month"], 36)

    def test_every_number_reaches_its_named_outcome_without_game_or_llm(self):
        cases = read_examples()
        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case["id"] for case in cases}), 10)
        for number, case in enumerate(cases, 1):
            with self.subTest(case=case["id"]):
                result = forecast(choose_example(str(number), cases)["game_result"])
                self.assertEqual(result["outcome"]["id"], case["id"])
                self.assertEqual(result["outcome"]["name"], case["name"])
                self.assertEqual(result["warnings"], [])

    def test_rejected_protocol_keeps_pledged_funds_separate_from_paid_funds(self):
        result = forecast(choose_example("Каждый сам за себя", read_examples())["game_result"])
        self.assertFalse(result["protocol"]["accepted"])
        self.assertEqual(sum(result["protocol"]["contributions"].values()), 270_000_000)
        self.assertEqual(result["paid"], 90_000_000)
        self.assertIsNone(result["commissioning_month"])

    def test_unknown_choice_shows_available_outcomes_instead_of_picking_one(self):
        for query in ("0", "11", "Несуществующий исход"):
            with self.subTest(query=query), self.assertRaisesRegex(ValueError, "1. Открытая река.*10. Застой"):
                choose_example(query, read_examples())

    def test_export_contains_current_examples_and_model_in_an_independent_graph(self):
        flow = json.loads(DEMO_FLOW_PATH.read_text(encoding="utf-8"))
        nodes = {node["data"]["type"]: node for node in flow["data"]["nodes"]}
        self.assertEqual(set(nodes), {"ChatInput", "ConsequencesExample", "ConsequencesModel", "ChatOutput"})
        self.assertEqual(nodes["ConsequencesExample"]["data"]["node"]["template"]["code"]["value"], render_selector())
        self.assertEqual(nodes["ConsequencesModel"]["data"]["node"]["template"]["code"]["value"],
                         COMPONENT_PATH.read_text(encoding="utf-8"))
        self.assertIn("проверочный пример", nodes["ChatOutput"]["data"]["node"]["template"]["sender_name"]["value"])
        self.assertEqual(flow, build_demo(json.loads(FLOW_PATH.read_text(encoding="utf-8")),
                                         nodes["ConsequencesExample"]["data"]["node"],
                                         nodes["ChatInput"]["data"]["node"]))
        links = {(edge["source"], edge["data"]["sourceHandle"]["name"], edge["target"],
                  edge["data"]["targetHandle"]["fieldName"]) for edge in flow["data"]["edges"]}
        by_type = {name: node["id"] for name, node in nodes.items()}
        self.assertEqual(links, {
            (by_type["ChatInput"], "message", by_type["ConsequencesExample"], "input_value"),
            (by_type["ConsequencesExample"], "game_result", by_type["ConsequencesModel"], "game_result"),
            (by_type["ConsequencesModel"], "newspaper", by_type["ChatOutput"], "input_value"),
        })


if __name__ == "__main__":
    unittest.main()
