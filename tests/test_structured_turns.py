"""Контракт хода и оркестрация без Langflow, сетевых вызовов и установки зависимостей.

Из исходника компилируются реальные схемы и методы. Подменяются только внешние
границы: Component, сообщения LangChain, Data и модель с include_raw=True.
"""

import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "langflow/custom_components/my_components/dialog.py"


class Message:
    type = "human"

    def __init__(self, content):
        self.content = content


class Data:
    def __init__(self, data, text=""):
        self.data, self.text = data, text


def load_component():
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(("lfx", "langchain")):
            continue
        if isinstance(node, ast.ClassDef) and node.name == "TwoLLMContractGame":
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in {"inputs", "outputs"} for t in item.targets))]
        nodes.append(node)
    namespace = dict(__name__="structured_turn_test", Component=object,
                     HumanMessage=Message, SystemMessage=Message, AIMessage=Message,
                     Data=Data, OutputParserException=type("OutputParserException", (ValueError,), {}))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


NS = load_component()
Game, Turn = NS["TwoLLMContractGame"], NS["PlayerTurn"]
IDS = ["city_administration", "industrial_holding", "regional_government",
       "city_residents", "environmental_ngo"]
NAMES = ["Администрация", "Промкомплекс", "Регион", "Жители", "Эко-НКО"]


def turn(**changes):
    payload = dict(speech="Публичная позиция.", own_contribution_millions=None,
                   own_levers={}, demands=[], responses=[], withdrawn_demands=[],
                   addressed_to=None, vote=None)
    payload.update(changes)
    return payload


def demand(**changes):
    payload = dict(target=IDS[1], demand_type="pledge", lever_id=None,
                   value=50, text="Взнос 50 млн", counter_to=None)
    payload.update(changes)
    return payload


class Model:
    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.calls, self.bindings = [], []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.bindings.append(kwargs)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        voting = any("ИДЕТ ЭТАП 3:" in message.content for message in messages)
        payload = self.replies.pop(0) if self.replies else turn(
            vote="for" if voting else None,
            own_contribution_millions=None if voting else 54)
        raw = SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
        try:
            return dict(raw=raw, parsed=self.schema.model_validate(payload), parsing_error=None)
        except ValidationError as exc:
            return dict(raw=raw, parsed=None, parsing_error=exc)


class StructuredTurnTests(unittest.TestCase):
    def setUp(self):
        self.game = Game()
        self.game._debug_calls = []
        self.game._debug_initial_state = None
        self.game._rag_audit = []
        self.game.use_rag = False
        self.game.settings = None
        self.game.rules_prompt = ""
        self.game.turns = 5
        self.game.players = [Data(dict(id=pid, name=name, player_name=name,
                                       money=200_000_000, budget=200_000_000,
                                       prompt="Представляй свою сторону.", llm=Model()))
                             for pid, name in zip(IDS, NAMES)]
        self.players, _, self.state, settings = self.game._init_session()
        self.conditions = settings["levers"]
        self.cost = settings["total_cost"]
        self.sleep = patch.object(NS["time"], "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def prepare(self, payload, speaker=NAMES[0], voting=False):
        return self.game._prepare_player_turn(Turn.model_validate(payload), speaker,
                                               self.state, self.conditions, self.cost, voting)

    def request(self, replies, voting=False):
        model = Model(replies)
        current = {**self.players[0], "llm": model}
        result = self.game._structured_turn(current, "Проверка", 1,
            [Message("Сделай ход")], self.state, self.conditions, self.cost, voting)
        return result, model

    def add_demand(self):
        self.state, _ = self.prepare(turn(demands=[demand()]))

    def test_nullable_and_zero_have_different_meanings(self):
        self.state["players"][NAMES[0]]["contribution"] = 10
        candidate, _ = self.prepare(turn())
        self.assertEqual(candidate["players"][NAMES[0]]["contribution"], 10)
        candidate, _ = self.prepare(turn(own_contribution_millions=0))
        self.assertEqual(candidate["players"][NAMES[0]]["contribution"], 0)

    def test_small_amounts_are_converted_exactly_once(self):
        candidate, _ = self.prepare(turn(own_contribution_millions=0.000001,
                                        demands=[demand(value=0.000002)]))
        self.assertEqual(candidate["players"][NAMES[0]]["contribution"], 1)
        self.assertEqual(candidate["demands"][0]["value"], 2)

    def test_whole_turn_rejected_without_mutation(self):
        before = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            self.prepare(turn(own_contribution_millions=30, own_levers={"plant_mode": "70%"}))
        self.assertEqual(self.state, before)

    def test_exact_player_ids_are_required(self):
        for payload in (turn(addressed_to="Промкомплекс"), turn(demands=[demand(target="unknown")])):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.prepare(payload)

    def test_invalid_condition_values_are_rejected(self):
        for value in (True, 0, 49, "24"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(turn(own_levers={"timeline": value}), speaker=NAMES[1])

    def test_budget_is_checked_without_clamping(self):
        with self.assertRaises(ValueError):
            self.prepare(turn(own_contribution_millions=201))
        with self.assertRaises(ValueError):
            self.prepare(turn(demands=[demand(value=201)]))

    def test_accept_requires_matching_explicit_commitment(self):
        self.add_demand()
        for amount in (None, 30):
            with self.subTest(amount=amount), self.assertRaises(ValueError):
                self.prepare(turn(own_contribution_millions=amount,
                    responses=[dict(demand_id="d1", decision="accept")]), speaker=NAMES[1])
        candidate, _ = self.prepare(turn(own_contribution_millions=50,
            responses=[dict(demand_id="d1", decision="accept")]), speaker=NAMES[1])
        self.assertEqual(candidate["demands"][0]["status"], "agreed")

    def test_only_recipient_can_respond_and_only_author_can_withdraw(self):
        self.add_demand()
        with self.assertRaises(ValueError):
            self.prepare(turn(responses=[dict(demand_id="d1", decision="reject")]))
        with self.assertRaises(ValueError):
            self.prepare(turn(withdrawn_demands=["d1"]), speaker=NAMES[1])
        for demand_id in ("all", "Промкомплекс", "d99"):
            with self.subTest(demand_id=demand_id), self.assertRaises(ValueError):
                self.prepare(turn(withdrawn_demands=[demand_id]))

    def test_withdrawal_does_not_reuse_ids(self):
        self.add_demand()
        candidate, _ = self.prepare(turn(withdrawn_demands=["d1"], demands=[demand(value=60)]))
        self.assertEqual([d["id"] for d in candidate["demands"]], ["d1", "d2"])
        self.assertEqual(candidate["demands"][0]["status"], "withdrawn")

    def test_counter_is_linked_and_original_stays_open(self):
        self.add_demand()
        payload = turn(responses=[dict(demand_id="d1", decision="counter")],
                       demands=[demand(target=IDS[0], value=30, counter_to="d1")])
        candidate, _ = self.prepare(payload, speaker=NAMES[1])
        self.assertEqual(candidate["demands"][0]["status"], "pending")
        self.assertEqual(candidate["demands"][1]["counter_to"], "d1")
        payload["demands"] = []
        with self.assertRaises(ValueError):
            self.prepare(payload, speaker=NAMES[1])

    def test_vote_phase_cannot_change_state(self):
        with self.assertRaises(ValueError):
            self.prepare(turn(vote="for"))
        with self.assertRaises(ValueError):
            self.prepare(turn(vote="for", own_contribution_millions=10), voting=True)
        for vote in ("for", "against", "abstain"):
            candidate, action = self.prepare(turn(vote=vote), voting=True)
            self.assertEqual(candidate, self.state)
            self.assertEqual(action["status"], "ok")

    def test_one_repair_commits_only_valid_turn(self):
        bad = turn(own_contribution_millions=30, own_levers={"plant_mode": "70%"})
        (answer, action), model = self.request([bad, turn(own_contribution_millions=40)])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.bindings, [dict(method="function_calling", include_raw=True)])
        self.assertIn("Исправь весь PlayerTurn", model.calls[1][-1].content)
        self.assertEqual(self.state["players"][NAMES[0]]["contribution"], 40_000_000)
        self.assertEqual(self.state["levers"]["plant_mode"], "100%")
        self.assertIsNotNone(answer)

    def test_repeated_schema_error_skips_turn_and_preserves_state(self):
        before = copy.deepcopy(self.state)
        (answer, action), model = self.request([turn(vote=2), turn(vote=2)])
        self.assertIsNone(answer)
        self.assertEqual(action["status"], "invalid_response")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.state, before)
        self.assertTrue(all("validation_error" in call for call in self.game._debug_calls))

    def test_full_game_uses_structured_players_without_converter(self):
        result = self.game.run_chat().data
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["voting_summary"]["votes_for"], 5)
        self.assertEqual(result["voting_summary"]["majority_threshold"], 3)
        self.assertTrue(all(h["structured_response"] for h in result["history"]))

    def test_addressed_speech_is_public_and_recipient_gets_next_turn(self):
        self.players[0]["llm"] = Model([turn(speech="Предложение региону.", addressed_to=IDS[2])])
        for player in self.state["players"].values():
            player["contribution"] = 54_000_000
        history = []
        self.game._run_phase2(self.players, self.state, history, "", 0, 2,
                              self.conditions, self.cost)
        self.assertEqual([h["speaker"] for h in history], [NAMES[0], NAMES[2]])
        self.assertIn("Предложение региону.", self.game._render_history(history))
        # Третий участник получает ту же публичную историю, а не приватную ветку.
        prompt = self.game._build_prompt_phase2(NAMES[3], "Роль", {}, self.state, history,
                                               "", 3, 5, 3, self.conditions, self.cost)
        self.assertIn("Предложение региону.", prompt)

    def test_last_call_uses_schema(self):
        history = []
        self.game._run_last_call(self.players, self.state, history, "", 0,
                                 self.conditions, self.cost)
        self.assertEqual(len(history), 5)
        self.assertTrue(all(h["is_last_call"] and h["structured_response"] for h in history))
        self.assertEqual(sum(p["contribution"] for p in self.state["players"].values()), self.cost)

    def test_withdrawn_demand_is_not_presented_as_pending(self):
        self.add_demand()
        self.state, _ = self.prepare(turn(withdrawn_demands=["d1"]))
        prompt = self.game._build_prompt_phase2(NAMES[1], "Роль", {}, self.state, [],
                                               "", 1, 5, 1, self.conditions, self.cost)
        self.assertNotIn("К ТЕБЕ ВЫДВИНУТЫ", prompt)
        self.assertIn("ОТОЗВАНО", prompt)

    def test_counter_to_unrelated_player_is_rejected(self):
        self.add_demand()
        with self.assertRaises(ValueError):
            self.prepare(turn(responses=[dict(demand_id="d1", decision="counter")],
                demands=[demand(target=IDS[2], counter_to="d1")]), speaker=NAMES[1])

    def test_condition_accept_requires_explicit_change(self):
        self.state, _ = self.prepare(turn(demands=[demand(demand_type="lever",
            lever_id="plant_mode", value="70%")]))
        response = [dict(demand_id="d1", decision="accept")]
        with self.assertRaises(ValueError):
            self.prepare(turn(responses=response), speaker=NAMES[1])
        candidate, _ = self.prepare(turn(responses=response, own_levers={"plant_mode": "70%"}),
                                    speaker=NAMES[1])
        self.assertEqual(candidate["levers"]["plant_mode"], "70%")

    def test_network_errors_do_not_turn_into_invalid_player_votes(self):
        model = Model()
        with patch.object(model, "invoke", side_effect=RuntimeError("API unavailable")):
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                self.game._structured_turn({**self.players[0], "llm": model}, "Торг", 1,
                    [Message("Ход")], self.state, self.conditions, self.cost)

    def test_exports_contain_same_component_and_have_no_dangling_edges(self):
        for path in (ROOT / "langflow").glob("Negotiation_MVP*.json"):
            with self.subTest(path=path):
                data = json.loads(path.read_text())["data"]
                ids = {node["id"] for node in data["nodes"]}
                component = next(node for node in data["nodes"]
                                 if node["data"]["type"] == "TwoLLMContractGame")
                template = component["data"]["node"]["template"]
                self.assertEqual(template["code"]["value"], SOURCE.read_text())
                self.assertNotIn("converter_model", template)
                for edge in data["edges"]:
                    self.assertIn(edge["source"], ids)
                    self.assertIn(edge["target"], ids)
                    self.assertNotEqual(edge["data"]["targetHandle"]["fieldName"], "converter_model")

    def test_invalid_vote_is_not_abstention_or_against(self):
        self.players[0]["llm"] = Model([turn(), turn()])
        result = self.game._run_phase3(self.players, self.state, [], 0,
                                      self.conditions, self.cost).data
        votes = result["voting_summary"]
        self.assertIsNone(votes["records"][0]["vote"])
        self.assertEqual(votes["records"][0]["status"], "invalid_response")
        self.assertEqual((votes["votes_for"], votes["votes_abstain"], votes["votes_against"],
                          votes["votes_invalid"], votes["majority_threshold"]), (4, 0, 0, 1, 3))


if __name__ == "__main__":
    unittest.main()
