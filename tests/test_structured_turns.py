"""Контракт хода и оркестрация без Langflow, сетевых вызовов и установки зависимостей.

Схемы и правила импортируются из пакета. Подменяются только внешние
границы: Component, сообщения LangChain, Data и модель с include_raw=True.
"""

import ast
import asyncio
import os
import tempfile
import __future__
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from negotiation_game.build_component import render_component
from negotiation_game.report import render_dialogue, render_history
from negotiation_game.rules import prepare_player_turn
from negotiation_game.schemas import PlayerTurn, normalize_player_turn, player_turn_tool_schema


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "langflow/custom_components/my_components/dialog.py"
IMPLEMENTATION = ROOT / "negotiation_game/langflow_component.py"


class Message:
    type = "human"

    def __init__(self, content):
        self.content = content


class Data:
    def __init__(self, data, text=""):
        self.data, self.text = data, text


def load_component(path=IMPLEMENTATION, postponed_annotations=False):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module.startswith(("lfx", "langchain")):
            continue
        if isinstance(node, ast.ClassDef) and node.name == "NegotiationGame":
            node.body = [item for item in node.body if not (
                isinstance(item, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in {"inputs", "outputs"} for t in item.targets))]
        nodes.append(node)
    namespace = dict(__name__="structured_turn_test", __package__="negotiation_game", Component=object,
                     HumanMessage=Message, SystemMessage=Message, AIMessage=Message,
                     Data=Data, OutputParserException=type("OutputParserException", (ValueError,), {}))
    flags = __future__.annotations.compiler_flag if postponed_annotations else 0
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec",
                 flags=flags, dont_inherit=True), namespace)
    return namespace


NS = load_component()
Game, Turn = NS["NegotiationGame"], PlayerTurn
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


def wire_turn(**changes):
    payload = dict(speech="Публичная позиция.",
                   own_contribution=dict(action="unchanged", amount_millions=0),
                   own_conditions=[], pledge_demands=[], condition_demands=[],
                   responses=[], withdrawn_demands=[], addressed_to="everyone", vote="none")
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

    async def ainvoke(self, messages):
        return self.invoke(messages)

    def invoke(self, messages):
        self.calls.append(list(messages))
        voting = any("ИДЕТ ЭТАП 3:" in message.content for message in messages)
        payload = self.replies.pop(0) if self.replies else turn(
            vote="for" if voting else None,
            own_contribution_millions=None if voting else 54)
        raw = SimpleNamespace(content="", tool_calls=[dict(name="PlayerTurn", args=payload)])
        if isinstance(self.schema, dict):
            return dict(raw=raw, parsed=payload, parsing_error=None)
        try:
            return dict(raw=raw, parsed=self.schema.model_validate(payload), parsing_error=None)
        except ValidationError as exc:
            return dict(raw=raw, parsed=None, parsing_error=exc)


class StructuredTurnTests(unittest.IsolatedAsyncioTestCase):
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
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.environment = patch.dict(os.environ, {"NEGOTIATION_GAMES_DIR": self.temp.name})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.sleep = patch.object(NS["asyncio"], "sleep", new_callable=AsyncMock)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def prepare(self, payload, speaker=NAMES[0], voting=False):
        return prepare_player_turn(Turn.model_validate(payload), speaker,
                                   self.state, self.conditions, self.cost, voting)

    async def request(self, replies, voting=False):
        model = Model(replies)
        current = {**self.players[0], "llm": model}
        result = await self.game._structured_turn(current, "Проверка", 1,
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

    async def test_one_repair_commits_only_valid_turn(self):
        bad = turn(own_contribution_millions=30, own_levers={"plant_mode": "70%"})
        (answer, action), model = await self.request([bad, turn(own_contribution_millions=40)])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.bindings, [dict(method="function_calling", include_raw=True)])
        self.assertIn("Исправь весь PlayerTurn", model.calls[1][-1].content)
        self.assertEqual(self.state["players"][NAMES[0]]["contribution"], 40_000_000)
        self.assertEqual(self.state["levers"]["plant_mode"], "100%")
        self.assertIsNotNone(answer)

    async def test_repeated_schema_error_skips_turn_and_preserves_state(self):
        before = copy.deepcopy(self.state)
        (answer, action), model = await self.request([turn(vote=2), turn(vote=2)])
        self.assertIsNone(answer)
        self.assertEqual(action["status"], "invalid_response")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.state, before)
        self.assertTrue(all("validation_error" in call for call in self.game._debug_calls))

    async def test_known_names_and_nullable_artifacts_are_accepted_and_audited(self):
        payload = turn(addressed_to="все", vote="null", own_contribution_millions="",
                       demands=[demand(target="Промкомплекс", lever_id="null", counter_to="")])
        original = copy.deepcopy(payload)
        (answer, action), model = await self.request([payload])
        self.assertIsNotNone(answer)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(payload, original)
        self.assertIsNone(answer.addressed_to)
        self.assertEqual(answer.demands[0].target, IDS[1])
        audit = self.game._debug_calls[0]
        self.assertEqual(audit["raw_arguments"], original)
        self.assertEqual(len(audit["normalization"]), 6)
        self.assertIsNone(audit["normalized_arguments"]["demands"][0]["counter_to"])
        self.assertEqual(self.state["demands"][0]["value"], 50_000_000)

    async def test_exact_name_address_can_be_normalized(self):
        (answer, _), model = await self.request([turn(addressed_to=NAMES[1])])
        self.assertEqual(answer.addressed_to, IDS[1])
        self.assertEqual(len(model.calls), 1)

    def test_normalization_does_not_guess_amounts_types_or_broken_ids(self):
        payload = turn(own_contribution_millions="50", addressed_to="всем",
                       demands=[demand(target="все", demand_type="pledge",
                                       lever_id="tariffs_freeze", value=True, counter_to="}, {")])
        normalized, changes = normalize_player_turn(payload, self.state["player_ids"])
        self.assertEqual(normalized, payload)
        self.assertEqual(changes, [])
        with self.assertRaises(ValidationError):
            Turn.model_validate(normalized)

    def test_ambiguous_names_do_not_resolve_to_an_arbitrary_player(self):
        normalized, changes = normalize_player_turn(turn(addressed_to="Общее имя"),
                                                    {"a": "Общее имя", "b": "Общее имя"})
        self.assertEqual(normalized["addressed_to"], "Общее имя")
        self.assertEqual(changes, [])

    def test_null_normalization_is_limited_to_nullable_fields(self):
        payload = turn(speech="null", demands=[demand(target="", value="null")],
                       responses=[dict(demand_id="null", decision="reject")],
                       withdrawn_demands=[""])
        normalized, changes = normalize_player_turn(payload, self.state["player_ids"])
        self.assertEqual(normalized, payload)
        self.assertEqual(changes, [])

    async def test_mixed_demand_is_repaired_without_applying_the_bad_amount(self):
        before = copy.deepcopy(self.state)
        bad = turn(own_contribution_millions=30,
                   demands=[demand(demand_type="pledge", lever_id="tax_relief", value=True)])
        (answer, _), model = await self.request([bad, turn()])
        self.assertIsNotNone(answer)
        self.assertEqual(self.state, before)
        repair = model.calls[1][-1].content
        self.assertIn('"loc": ["demands", 0]', repair)
        self.assertIn('"lever_id": "tax_relief"', repair)
        self.assertIn('"condition_demands":', repair)
        self.assertNotIn("namespace(", repair)
        self.assertNotIn("response_metadata", repair)

    async def test_bad_counter_to_is_not_silently_deleted(self):
        before = copy.deepcopy(self.state)
        bad = turn(own_contribution_millions=30, demands=[demand(counter_to="}, {")])
        (answer, action), model = await self.request([bad, bad])
        self.assertIsNone(answer)
        self.assertEqual(action["status"], "invalid_response")
        self.assertEqual(self.state, before)

    async def test_unknown_target_is_not_replaced_with_everyone(self):
        before = copy.deepcopy(self.state)
        bad = turn(demands=[demand(target="все")])
        (answer, _), model = await self.request([bad, bad])
        self.assertIsNone(answer)
        self.assertEqual(self.state, before)

    async def test_vote_artifacts_can_be_normalized_without_changing_commitments(self):
        before = copy.deepcopy(self.state)
        (answer, action), model = await self.request(
            [turn(addressed_to="все", own_contribution_millions="null", vote="for")], voting=True)
        self.assertIsNotNone(answer)
        self.assertEqual(action["vote"], "ЗА")
        self.assertEqual(self.state, before)

    def test_tool_arguments_recovered_before_pydantic_validation(self):
        payload = turn(addressed_to="все")
        raw = SimpleNamespace(tool_calls=[dict(name="PlayerTurn", args=payload)])
        self.assertEqual(self.game._player_turn_arguments(
            dict(raw=raw, parsed=None, parsing_error=ValueError("parse"))), payload)

    def test_invalid_tool_json_or_multiple_calls_are_not_salvaged(self):
        for calls in ([dict(name="OtherTool", args=turn())],
                      [dict(name="PlayerTurn", args=turn())] * 2,
                      [dict(name="PlayerTurn", args='{"speech":')]):
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                self.game._player_turn_arguments(dict(raw=SimpleNamespace(tool_calls=calls)))

    def test_plain_speech_is_not_interpreted_as_a_tool_call(self):
        with self.assertRaises(ValueError):
            self.game._player_turn_arguments(dict(raw=SimpleNamespace(content=json.dumps(turn()))))

    def test_transport_schema_limits_role_ids_and_separates_demand_types(self):
        schema = player_turn_tool_schema(self.state["player_ids"])
        self.assertEqual(schema["properties"]["addressed_to"]["enum"], [*IDS, "everyone"])
        pledge = schema["$defs"]["PledgeDemandInput"]["properties"]
        lever = schema["$defs"]["ConditionDemandInput"]["properties"]
        self.assertNotIn("lever_id", pledge)
        self.assertEqual(pledge["amount_millions"]["type"], "number")
        self.assertNotIn("amount_millions", lever)
        self.assertEqual(lever["target"]["enum"], IDS)
        self.assertEqual(schema["properties"]["vote"]["enum"], ["none"])
        voting = player_turn_tool_schema(self.state["player_ids"], voting=True)
        self.assertEqual(voting["properties"]["addressed_to"]["enum"], ["everyone"])
        self.assertEqual(voting["properties"]["pledge_demands"]["maxItems"], 0)
        self.assertEqual(voting["properties"]["vote"]["enum"], ["for", "against", "abstain"])
        self.assertEqual(schema["title"], "PlayerTurn")
        # Альтернативы типов теряются в установленном Gpt2Giga.
        def check(node):
            if isinstance(node, dict):
                self.assertTrue(set(node).isdisjoint({"anyOf", "oneOf", "allOf"}))
                self.assertNotEqual(node.get("type"), "null")
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for value in node:
                    check(value)
        check(schema)

    async def test_wire_response_applies_typed_conditions_and_money(self):
        payload = wire_turn(own_contribution=dict(action="set", amount_millions=20),
            own_conditions=[dict(lever_id="tariffs_freeze", value_json="true")],
            pledge_demands=[dict(target=IDS[1], amount_millions=50, text="Взнос 50 млн", counter_to="")],
            condition_demands=[dict(target=IDS[1], lever_id="plant_mode", value_json='"70%"',
                                    text="Снизить мощность", counter_to="")])
        (answer, action), model = await self.request([payload])
        self.assertIsNotNone(answer)
        self.assertEqual(self.state["players"][NAMES[0]]["contribution"], 20_000_000)
        self.assertIs(self.state["levers"]["tariffs_freeze"], True)
        self.assertEqual(self.state["demands"][0]["value"], 50_000_000)
        self.assertEqual(self.state["demands"][1]["value"], "70%")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.game._debug_calls[0]["raw_arguments"], payload)

    async def test_wire_zero_is_distinct_from_unchanged(self):
        self.state["players"][NAMES[0]]["contribution"] = 10_000_000
        await self.request([wire_turn()])
        self.assertEqual(self.state["players"][NAMES[0]]["contribution"], 10_000_000)
        await self.request([wire_turn(own_contribution=dict(action="set", amount_millions=0))])
        self.assertEqual(self.state["players"][NAMES[0]]["contribution"], 0)

    async def test_wire_null_artifacts_and_exact_names_are_normalized(self):
        payload = wire_turn(addressed_to="все", vote="null", pledge_demands=[
            dict(target=NAMES[1], amount_millions=10, text="Взнос", counter_to="null")])
        (answer, _), model = await self.request([payload])
        self.assertIsNotNone(answer)
        self.assertIsNone(answer.addressed_to)
        self.assertIsNone(answer.vote)
        self.assertEqual(answer.demands[0].target, IDS[1])
        self.assertIsNone(answer.demands[0].counter_to)
        self.assertEqual(self.game._debug_calls[-1]["raw_arguments"], payload)
        self.assertEqual(len(model.calls), 1)

    async def test_wire_vote_has_no_financial_side_effects(self):
        before = copy.deepcopy(self.state)
        (answer, action), model = await self.request([wire_turn(vote="for")], voting=True)
        self.assertEqual(action["vote"], "ЗА")
        self.assertEqual(self.state, before)
        self.assertEqual(len(model.calls), 1)

    async def test_wire_invalid_values_and_mixed_formats_are_not_applied(self):
        for bad in (
            wire_turn(own_contribution=dict(action="unchanged", amount_millions=40)),
            wire_turn(own_contribution=dict(action="set", amount_millions=True)),
            wire_turn(own_conditions=[dict(lever_id="tariffs_freeze", value_json="True")]),
            wire_turn(own_conditions=[dict(lever_id="tariffs_freeze", value_json="1")]),
            wire_turn(own_conditions=[dict(lever_id="tariffs_freeze", value_json="true")] * 2),
            wire_turn(own_contribution_millions=30),
            wire_turn(vote="for"),
        ):
            with self.subTest(bad=bad):
                before = copy.deepcopy(self.state)
                (answer, action), model = await self.request([bad, bad])
                self.assertIsNone(answer)
                self.assertEqual(self.state, before)

    async def test_wire_repair_includes_original_arguments_and_field_errors(self):
        bad = wire_turn(own_contribution=dict(action="set", amount_millions=True))
        (answer, action), model = await self.request([bad, wire_turn()])
        self.assertIsNotNone(answer)
        repair = model.calls[1][-1].content
        self.assertIn('"loc": ["own_contribution", "amount_millions"]', repair)
        self.assertIn('"amount_millions": true', repair)
        self.assertNotIn("response_metadata", repair)
        self.assertEqual(self.game._debug_calls[-2]["raw_arguments"], bad)

    async def test_failed_decode_preserves_original_and_normalization_audit(self):
        bad = wire_turn(addressed_to="все", own_contribution=dict(action="set", amount_millions=True))
        await self.request([bad, bad])
        audit = self.game._debug_calls[-1]
        self.assertEqual(audit["raw_arguments"], bad)
        self.assertEqual(audit["normalization"], [dict(field="addressed_to", before="все", after="everyone")])

    async def test_full_game_uses_structured_players_without_converter(self):
        result = (await self.game.run_chat()).data
        self.assertTrue(result["consensus_reached"])
        self.assertEqual(result["voting_summary"]["votes_for"], 5)
        self.assertEqual(result["voting_summary"]["majority_threshold"], 3)
        self.assertTrue(all(h["structured_response"] for h in result["history"]))

    async def test_addressed_speech_is_public_and_recipient_gets_next_turn(self):
        self.players[0]["llm"] = Model([turn(speech="Предложение региону.", addressed_to=IDS[2])])
        for player in self.state["players"].values():
            player["contribution"] = 54_000_000
        history = []
        await self.game._run_phase2(self.players, self.state, history, "", 0, 2,
                              self.conditions, self.cost)
        self.assertEqual([h["speaker"] for h in history], [NAMES[0], NAMES[2]])
        self.assertIn("Предложение региону.", render_history(history))
        # Третий участник получает ту же публичную историю, а не приватную ветку.
        prompt = self.game._build_prompt_phase2(NAMES[3], "Роль", {}, self.state, history,
                                               "", 3, 5, 3, self.conditions, self.cost)
        self.assertIn("Предложение региону.", prompt)

    async def test_last_call_uses_schema(self):
        history = []
        await self.game._run_last_call(self.players, self.state, history, "", 0,
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

    async def test_network_errors_do_not_turn_into_invalid_player_votes(self):
        model = Model()
        with patch.object(model, "invoke", side_effect=RuntimeError("API unavailable")):
            with self.assertRaisesRegex(RuntimeError, "API unavailable"):
                await self.game._structured_turn({**self.players[0], "llm": model}, "Торг", 1,
                    [Message("Ход")], self.state, self.conditions, self.cost)

    def test_exports_contain_same_component_and_have_no_dangling_edges(self):
        self.assertEqual(SOURCE.read_text(), render_component())
        for path in (ROOT / "langflow").glob("Negotiation_MVP*.json"):
            with self.subTest(path=path):
                data = json.loads(path.read_text())["data"]
                ids = {node["id"] for node in data["nodes"]}
                component = next(node for node in data["nodes"]
                                 if node["data"]["type"] == "NegotiationGame")
                self.assertEqual(component["id"], "NegotiationGame-main")
                template = component["data"]["node"]["template"]
                self.assertEqual(template["code"]["value"], SOURCE.read_text())
                self.assertNotIn("converter_model", template)
                for edge in data["edges"]:
                    self.assertIn(edge["source"], ids)
                    self.assertIn(edge["target"], ids)
                    self.assertEqual(edge["data"]["sourceHandle"]["id"], edge["source"])
                    self.assertEqual(edge["data"]["targetHandle"]["id"], edge["target"])
                    for side in ("source", "target"):
                        handle = json.loads(edge[side + "Handle"].replace("œ", '"'))
                        self.assertEqual(handle["id"], edge[side])
                    self.assertNotEqual(edge["data"]["targetHandle"]["fieldName"], "converter_model")

    async def test_bundled_component_matches_package_game(self):
        bundled = load_component(SOURCE)
        game = bundled["NegotiationGame"]()
        game.__dict__.update(copy.deepcopy(self.game.__dict__))
        left, right = (await game.run_chat()).data, (await self.game.run_chat()).data
        for payload in (left, right):
            payload.pop("run_id", None)
            payload.pop("files", None)
        self.assertEqual(left, right)

    def test_dynamic_component_resolves_postponed_schema_annotations(self):
        dynamic = load_component(SOURCE, postponed_annotations=True)
        schema = dynamic["player_turn_schema"]()
        self.assertTrue(schema.__pydantic_complete__)
        self.assertEqual(set(schema.model_json_schema()["properties"]), set(turn()))
        payload = turn(demands=[demand()], responses=[dict(demand_id="d1", decision="reject")])
        answer = schema.model_validate(payload)
        self.assertEqual(answer.demands[0].value, 50)
        self.assertEqual(answer.responses[0].decision, "reject")
        with self.assertRaises(ValidationError):
            schema.model_validate(turn(demands=[demand(value=True)]))
        self.assertIs(dynamic["player_turn_schema"](), schema)

    async def test_dynamic_component_runs_game_with_postponed_annotations(self):
        dynamic = load_component(SOURCE, postponed_annotations=True)
        game = dynamic["NegotiationGame"]()
        game.__dict__.update(copy.deepcopy(self.game.__dict__))
        left, right = (await game.run_chat()).data, (await self.game.run_chat()).data
        for payload in (left, right):
            payload.pop("run_id", None)
            payload.pop("files", None)
        self.assertEqual(left, right)

    async def test_report_rendering_does_not_change_game_data(self):
        result = (await self.game.run_chat()).data
        arguments = dict(history=result["history"], state=result["state"],
                         voting_summary=result["voting_summary"], total_cost=self.cost,
                         levers_config=self.conditions)
        before = copy.deepcopy(arguments)
        report = render_dialogue(**arguments)
        self.assertEqual(arguments, before)
        self.assertIn("ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ", report)
        self.assertIn("СТЕНОГРАММА", report)

    async def test_invalid_vote_is_not_abstention_or_against(self):
        self.players[0]["llm"] = Model([turn(), turn()])
        result = (await self.game._run_phase3(self.players, self.state, [], 0,
                                       self.conditions, self.cost)).data
        votes = result["voting_summary"]
        self.assertIsNone(votes["records"][0]["vote"])
        self.assertEqual(votes["records"][0]["status"], "invalid_response")
        self.assertIn("ГОЛОС НЕ ПОЛУЧЕН", result["report"])
        self.assertEqual((votes["votes_for"], votes["votes_abstain"], votes["votes_against"],
                          votes["votes_invalid"], votes["majority_threshold"]), (4, 0, 0, 1, 3))

    def checkpoint(self):
        path = next(Path(self.temp.name).glob("*/result.json"))
        return json.loads(path.read_text())

    async def test_network_failure_returns_and_persists_partial_report(self):
        broken = self.game.players[1].data["llm"]
        with patch.object(broken, "ainvoke", side_effect=ConnectionError("Прокси недоступен")) as invoke:
            result = (await self.game.run_chat()).data
        self.assertEqual(invoke.await_count, 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(result["history"]), 1)
        self.assertIsNone(result["consensus_reached"])
        self.assertIn("Игра прервана", result["report"])
        self.assertNotIn("ПРОТОКОЛ НЕ ПРИНЯТ", result["report"])
        self.assertEqual(self.checkpoint()["status"], "failed")
        self.assertTrue(Path(result["files"]["result.md"]).exists())

    async def test_cancellation_stops_call_and_saves_report(self):
        started = asyncio.Event()
        stopped = asyncio.Event()
        calls = []

        async def blocked(messages):
            calls.append(messages)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        model = self.game.players[0].data["llm"]
        with patch.object(model, "ainvoke", side_effect=blocked):
            task = asyncio.create_task(self.game.run_chat())
            await asyncio.wait_for(started.wait(), 1)
            self.assertEqual(self.checkpoint()["status"], "running")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(stopped.is_set())
            self.assertEqual(len(calls), 1)
        self.assertEqual(self.checkpoint()["status"], "cancelled")
        self.assertIn("Игра отменена", self.checkpoint()["report"])

    async def test_request_timeout_has_bounded_attempts(self):
        self.game.request_timeout_seconds = 0.01

        async def blocked(messages):
            await asyncio.Event().wait()

        model = self.game.players[0].data["llm"]
        with patch.object(model, "ainvoke", side_effect=blocked) as invoke:
            result = (await self.game.run_chat()).data
        self.assertEqual(invoke.await_count, 2)
        self.assertEqual(result["status"], "failed")
        self.assertIn("не ответила", result["error"])
        self.assertEqual(self.checkpoint()["status"], "failed")

    async def test_game_timeout_cancels_pending_request(self):
        self.game.game_timeout_seconds = 0.01

        async def blocked(messages):
            await asyncio.Event().wait()

        model = self.game.players[0].data["llm"]
        with patch.object(model, "ainvoke", side_effect=blocked) as invoke:
            result = (await self.game.run_chat()).data
        self.assertEqual(invoke.await_count, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("общий лимит", result["error"])

    async def test_successful_game_is_persisted_as_completed(self):
        result = (await self.game.run_chat()).data
        saved = self.checkpoint()
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["history"], result["history"])
        self.assertTrue(saved["consensus_reached"])
        self.assertEqual(Path(result["files"]["result.md"]).read_text().rstrip(), result["report"])


if __name__ == "__main__":
    unittest.main()
