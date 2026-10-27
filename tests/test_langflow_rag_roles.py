"""Verify retrieval filters in both shipped copies of the Langflow component."""

import ast
import base64
import copy
import json
import sys
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PLAYERS = (
    ("city_administration", "Администрация", "admin"),
    ("industrial_holding", "Промкомплекс", "holding"),
    ("regional_government", "Регион", "region"),
    ("city_residents", "Жители", "residents"),
    ("environmental_ngo", "Эко-НКО", "ngo"),
)
RECORDS = [json.loads(line) for line in (ROOT / "full_embeddings.jsonl").read_text().splitlines() if line.strip()]
CORPUS_B64 = base64.b64encode(zlib.compress("\n".join(
    json.dumps({key: value for key, value in record.items() if key != "embedding"})
    for record in RECORDS).encode())).decode()
SEARCH_CORPUS_B64 = base64.b64encode(zlib.compress((zlib.decompress(
    base64.b64decode(CORPUS_B64)).decode() + "\n" + json.dumps({
        "chunk_id": "EXTRA-TEST", "text": "Дополнительная справка",
        "area": "external_case", "index_policy": "main_index",
        "audience": [role for _, _, role in PLAYERS],
    })).encode())).decode()


def component_versions():
    graph = json.loads((ROOT / "langflow/Negotiation_MVP_5_players.json").read_text())
    node = next(n for n in graph["data"]["nodes"] if n["data"]["type"] == "TwoLLMContractGame")
    return {
        "source": (ROOT / "langflow/custom_components/my_components/dialog.py").read_text(),
        "export": node["data"]["node"]["template"]["code"]["value"],
    }


def load_methods(code):
    # Compile actual production methods, omitting only Langflow UI dependencies.
    tree = ast.parse(code)
    selected = [n for n in tree.body if
                (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 {"RAG_ROLE_BY_PLAYER_ID", "DEFAULT_PROJECT_COST"} for t in n.targets))
                or (isinstance(n, ast.FunctionDef) and n.name in {
                    "_rag_role", "_get_default_settings", "_rag_chunk_text", "_format_money", "search_enabled",
                    "_extract_text", "_clean_command", "_clean_speech_message"})]
    game = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TwoLLMContractGame")
    selected.extend(n for n in game.body if isinstance(n, ast.FunctionDef)
                    and n.name in {"_rag_context", "_init_session", "_run_phase1", "_run_phase2",
                                   "_run_last_call", "_run_phase3"})
    namespace = {"copy": copy, "base64": base64, "json": json, "zlib": zlib,
                 "re": __import__("re"), "SystemMessage": SimpleNamespace,
                 "HumanMessage": SimpleNamespace, "Data": SimpleNamespace}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<Langflow RAG methods>", "exec"), namespace)
    return namespace


class LangflowRAGRoleTests(unittest.TestCase):
    def test_retrieval_uses_id_even_after_rename(self):
        for version, code in component_versions().items():
            methods = load_methods(code)
            for player_id, name, role in PLAYERS:
                for display_name in (name, "Произвольное имя"):
                    with self.subTest(version=version, player_id=player_id, name=display_name):
                        collection = Mock()
                        held = [r for r in RECORDS if "index_policy" in r]
                        collection.query.return_value = {
                            "ids": [["F-10", "EXTRA-TEST", "UNKNOWN", *[r["chunk_id"] for r in held]]],
                            "documents": [["Дубликат обязательного факта", "Дополнительная справка", "Неизвестный источник",
                                           *[r["text"] for r in held]]],
                            "metadatas": [[{"area": "shared"}, {"area": "external_case", "index_policy": "main_index"},
                                           {"area": "external_case", "index_policy": "main_index"},
                                           *[{**r, "search_enabled": True} for r in held]]],
                        }
                        database = Mock()
                        database.get_collection.return_value = collection
                        api = Mock()
                        api.embeddings.create.return_value = SimpleNamespace(
                            data=[SimpleNamespace(embedding=[0.1, 0.2])])
                        modules = {
                            "chromadb": SimpleNamespace(PersistentClient=Mock(return_value=database)),
                            "chromadb.config": SimpleNamespace(Settings=Mock()),
                            "openai": SimpleNamespace(OpenAI=Mock(return_value=api)),
                        }
                        game = SimpleNamespace(rag_config=SimpleNamespace(data={
                            "embedding_base_url": "http://unused", "embedding_api_key": "test",
                            "embedding_model": "test", "dimension": 2, "chroma_path": "unused",
                            "collection_name": "test", "top_k": 2,
                            "corpus_b64": SEARCH_CORPUS_B64,
                        }))
                        with patch.dict(sys.modules, modules):
                            text = methods["_rag_context"](game, display_name,
                                                           {"id": player_id, "role_id": "wrong"}, [], 1)
                        self.assertEqual(collection.query.call_args.kwargs["where"], {
                            "$and": [{"visible_" + role: True}, {"area": {"$nin": ["shared", "role"]}},
                                     {"search_enabled": True}]})
                        self.assertIn("[EXTRA-TEST]", text)
                        self.assertNotIn("Дубликат обязательного факта", text)
                        self.assertEqual(game._rag_audit[0]["chunk_ids"], ["EXTRA-TEST"])
                        self.assertEqual(game._rag_audit[0]["retrieval_status"], "ok")
                        for record in held:
                            self.assertNotIn(f"[{record['chunk_id']}]", text)
                        self.assertNotIn("[UNKNOWN]", text)

    def test_full_core_survives_search_failure_and_excludes_other_roles(self):
        for version, code in component_versions().items():
            methods = load_methods(code)
            for player_id, name, role in PLAYERS:
                with self.subTest(version=version, role=role):
                    game = SimpleNamespace(rag_config=SimpleNamespace(data={"corpus_b64": SEARCH_CORPUS_B64}))
                    with patch.dict(sys.modules, {"chromadb": None}):
                        text = methods["_rag_context"](game, name, {"id": player_id}, [], 1)
                    expected = [record for record in RECORDS
                                if record.get("area") in {"shared", "role"} and role in record["audience"]]
                    self.assertEqual(game._rag_audit[0]["core_chunk_ids"], [r["chunk_id"] for r in expected])
                    for record in expected:
                        self.assertIn(record["text"], text)
                    for record in RECORDS:
                        if record.get("area") == "role" and role not in record["audience"]:
                            self.assertNotIn(f"[{record['chunk_id']}]", text)
                    self.assertEqual(game._rag_audit[0]["retrieval_status"], "error")

    def test_missing_or_corrupt_core_is_an_explicit_error(self):
        for version, code in component_versions().items():
            for payload in (None, "not a compressed corpus", base64.b64encode(zlib.compress(b"[]")).decode()):
                with self.subTest(version=version, payload=payload):
                    game = SimpleNamespace(rag_config=SimpleNamespace(data={"corpus_b64": payload}))
                    with self.assertRaisesRegex(ValueError, "RAG:"):
                        load_methods(code)["_rag_context"](game, "Жители", {"id": "city_residents"}, [], 1)

    def test_core_is_delivered_at_every_stage_with_no_search_in_final_stages(self):
        for version, code in component_versions().items():
            with self.subTest(version=version):
                methods = load_methods(code)
                llm = Mock()
                llm.invoke.return_value = SimpleNamespace(content="Без изменений")
                player = {"name": "Жители", "prompt": "Стратегия", "llm": llm,
                          "profile": {"id": "city_residents", "money": 30_000_000}}
                game = SimpleNamespace(
                    rag_config=SimpleNamespace(data={"corpus_b64": CORPUS_B64}),
                    _build_prompt_phase1=Mock(return_value="ВСТУПЛЕНИЕ"),
                    _build_prompt_phase2=Mock(return_value="ТОРГ"),
                    _build_prompt_last_call=Mock(return_value="LAST CALL"),
                    _build_prompt_phase3=Mock(return_value="ГОЛОСОВАНИЕ"),
                    _parse=Mock(return_value={}),
                    _apply_actions=Mock(return_value={"message": "Без изменений"}),
                    _parse_vote=Mock(return_value=("ПРОТИВ", "Дефицит")),
                    _extract_addressed_player=Mock(return_value=None),
                    _render_dialogue=Mock(return_value="Отчёт"),
                    _invoke_model=lambda current, phase, turn, messages: current["llm"].invoke(messages),
                )
                game._rag_context = lambda *args, **kwargs: methods["_rag_context"](game, *args, **kwargs)
                state = {"players": {"Жители": {"contribution": 0, "share": 0}}, "levers": {}}
                history = []
                with patch.dict(sys.modules, {"chromadb": None}):
                    turn = methods["_run_phase1"](game, [player], state, history, "", 0, [], 270_000_000)
                    state["players"]["Жители"]["contribution"] = 270_000_000
                    turn = methods["_run_phase2"](game, [player], state, history, "", turn, 1, [], 270_000_000)
                state["players"]["Жители"]["contribution"] = 0
                turn = methods["_run_last_call"](game, [player], state, history, "", turn, [], 270_000_000)
                methods["_run_phase3"](game, [player], state, history, turn, [], 270_000_000)
                self.assertEqual(llm.invoke.call_count, 4)
                for call in llm.invoke.call_args_list:
                    prompt = call.args[0][1].content
                    self.assertIn("[F-10]", prompt)
                    self.assertIn("[RES-05]", prompt)
                    self.assertIn("[RES-08]", prompt)
                    self.assertNotIn("[HLD-01]", prompt)
                self.assertEqual([entry["retrieval_status"] for entry in game._rag_audit],
                                 ["skipped", "skipped", "skipped", "skipped"])
                self.assertEqual(game._rag_audit[0]["skip_reason"], "no_allowed_sources")

    def test_unknown_id_fails_before_retrieval_and_session_start(self):
        for version, code in component_versions().items():
            methods = load_methods(code)
            for player_id in (None, "unknown", [], "holding"):
                with self.subTest(version=version, player_id=player_id):
                    profile = {"id": player_id, "name": "Жители", "prompt": "Стратегия", "llm": Mock()}
                    game = SimpleNamespace(players=[profile], rag_config=SimpleNamespace(data={"enabled": True}))
                    with self.assertRaisesRegex(ValueError, "Неизвестная RAG-роль.*Жители"):
                        methods["_init_session"](game)
                    with self.assertRaisesRegex(ValueError, "Неизвестная RAG-роль"):
                        methods["_rag_context"](game, "Жители", profile, [], 1)
                    profile["llm"].invoke.assert_not_called()

    def test_disabled_rag_does_not_require_known_id(self):
        for version, code in component_versions().items():
            methods = load_methods(code)
            for config, enabled in ((None, True), (SimpleNamespace(data={"enabled": True}), False)):
                with self.subTest(version=version, enabled=enabled):
                    game = SimpleNamespace(
                        players=[{"id": "custom", "name": "Игрок", "prompt": "Стратегия", "llm": Mock()}],
                        rag_config=config, use_rag=enabled,
                    )
                    players, _, _, _ = methods["_init_session"](game)
                    self.assertEqual(players[0]["profile"]["id"], "custom")
                    self.assertEqual(methods["_rag_context"](game, "Игрок", {}, [], 1), "")
