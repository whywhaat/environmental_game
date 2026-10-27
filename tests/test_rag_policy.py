import ast
import base64
import hashlib
import json
import math
import os
import sys
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from negotiation_mvp.rag import RAGRetriever
from negotiation_mvp.rag_policy import search_enabled


ROOT = Path(__file__).resolve().parents[1]
RECORDS = [json.loads(line) for line in (ROOT / "full_embeddings.jsonl").read_text().splitlines() if line.strip()]
HELD = [r for r in RECORDS if r.get("index_policy") in
        {"reference_hold_not_main_index", "analogy_reference_hold"}]


class RAGPolicyTests(unittest.TestCase):
    def test_policy_rejects_all_held_and_unknown_material(self):
        self.assertEqual(len(HELD), 74)
        self.assertTrue(all(not search_enabled(r) for r in HELD))
        self.assertEqual(sum(search_enabled(r) for r in RECORDS), 85)
        for policy in (None, "unknown", "", "reference_hold_not_main_index", "analogy_reference_hold"):
            self.assertFalse(search_enabled({"area": "shared", "index_policy": policy, "search_enabled": True}))
        self.assertFalse(search_enabled({"area": "external_case"}))
        self.assertTrue(search_enabled({"area": "external_case", "index_policy": "main_index"}))

    def test_fallback_rejects_held_material_and_unknown_roles(self):
        for role in ("admin", "holding", "region", "residents", "ngo", "judge", "unknown"):
            with self.subTest(role=role):
                retriever = RAGRetriever({"top_k": 8})
                retriever._records = RECORDS
                with patch.dict(os.environ, {"OPENAI_API_KEY": "", "TSARROUTER_API_KEY": ""}):
                    chunks = retriever.retrieve(role, 1, "Норильск экологическое право концессия")
                self.assertTrue(all(search_enabled(r) and role in r["audience"] for r in chunks))
                if role == "unknown":
                    self.assertEqual(chunks, [])
                retriever._records = HELD
                self.assertEqual(retriever.retrieve(role, 1, "Норильск"), [])

    def test_python_search_rejects_stale_results_and_falls_back_on_failure(self):
        collection = Mock()
        # Simulate an old collection ignoring the new filter and returning held entries.
        collection.query.return_value = {
            "ids": [[r["chunk_id"] for r in HELD]], "documents": [[r["text"] for r in HELD]],
            "metadatas": [[{**r, "search_enabled": True, "index_policy": "main_index"} for r in HELD]],
        }
        database = Mock()
        database.get_collection.return_value = collection
        api = Mock()
        api.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1])])
        modules = {
            "chromadb": SimpleNamespace(PersistentClient=Mock(return_value=database)),
            "chromadb.config": SimpleNamespace(Settings=Mock()),
            "openai": SimpleNamespace(OpenAI=Mock(return_value=api)),
        }
        retriever = RAGRetriever({"embedding_api_key": "test", "chroma_path": "unused", "collection_name": "old"})
        retriever._records = RECORDS
        with patch.dict(sys.modules, modules):
            for fails in (False, True):
                with self.subTest(fails=fails):
                    collection.query.side_effect = RuntimeError("offline") if fails else None
                    chunks = retriever.retrieve("city_residents", 1, "Норильск")
                    self.assertTrue(chunks)
                    self.assertTrue(all(search_enabled(r) and "residents" in r["audience"] for r in chunks))
        self.assertEqual(collection.query.call_args.kwargs["where"], {
            "$and": [{"visible_residents": True}, {"search_enabled": True}]})

    def test_builder_versions_metadata_and_embedded_policies_match(self):
        graph = json.loads((ROOT / "langflow/Negotiation_MVP_5_players.json").read_text())
        sources = {"source": (ROOT / "langflow/custom_components/my_components/rag.py").read_text()}
        for node in graph["data"]["nodes"]:
            if node["data"]["type"] == "SharedProjectRAG":
                sources["export"] = node["data"]["node"]["template"]["code"]["value"]
        for version, code in sources.items():
            with self.subTest(version=version):
                tree = ast.parse(code)
                functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "search_enabled"]
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SharedProjectRAG")
                method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build_rag")
                constants = [n for n in tree.body if isinstance(n, ast.Assign)]
                namespace = dict(base64=base64, hashlib=hashlib, json=json, math=math, os=os,
                                 zlib=zlib, Path=Path, Data=SimpleNamespace, SecretStr=lambda value: value)
                exec(compile(ast.Module(body=[*constants, *functions, method], type_ignores=[]), "<RAG builder>", "exec"), namespace)
                self.assertEqual([namespace["search_enabled"](r) for r in RECORDS],
                                 [search_enabled(r) for r in RECORDS])
                collection = Mock()
                collection.count.return_value = 0
                database = Mock()
                database.get_or_create_collection.return_value = collection
                modules = {"chromadb": SimpleNamespace(PersistentClient=Mock(return_value=database)),
                           "chromadb.config": SimpleNamespace(Settings=Mock())}
                with patch.dict(sys.modules, modules):
                    output = namespace["build_rag"](SimpleNamespace(
                        persist_directory="/tmp/unused-rag-test", embedding_api_key="test",
                        embedding_base_url="http://unused", top_k=2))
                digest = hashlib.sha256(zlib.decompress(base64.b64decode(namespace["CORPUS_B64"])) ).hexdigest()
                self.assertNotEqual(output.data["collection_name"], "city_n_" + digest[:16])
                metadata = database.get_or_create_collection.call_args.kwargs["metadata"]
                self.assertEqual(metadata["index_schema_version"], namespace["INDEX_SCHEMA_VERSION"])
                indexed = {cid: meta for call in collection.upsert.call_args_list
                           for cid, meta in zip(call.kwargs["ids"], call.kwargs["metadatas"])}
                self.assertEqual(len(indexed), 159)
                self.assertEqual(sum(meta["search_enabled"] for meta in indexed.values()), 85)
                self.assertTrue(all(indexed[r["chunk_id"]]["search_enabled"] is False for r in HELD))
