"""Контекст для модели: факты и ограничения без технической обёртки RAG."""

import base64
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
import zlib

import test_structured_turns as structured


class RagContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        structured.StructuredTurnTests.setUp(self)
        self.game.use_rag = True
        self.game.rules_prompt = "Правило заседания: RULE_ONLY_IN_SYSTEM."
        self.records = [self.record("shared-id", "shared", ["admin", "holding", "region", "residents", "ngo"],
                                    "SHARED_FACT: смета проекта 270 млн рублей.")]
        for role in ("admin", "holding", "region", "residents", "ngo"):
            self.records.append(self.record(f"{role}-id", "role", [role],
                f"ROLE_FACT_{role}: соблюдай лимит своей стороны; см. J-04."))
        self.configure()

    @staticmethod
    def record(chunk_id, area, audience, body, policy=None):
        title, section, note = "Каталоговый заголовок", "Каталоговый раздел", "PRIVATE_AUTHORITY_NOTE"
        result = dict(chunk_id=chunk_id, area=area, audience=audience,
                      document_title=title, section_title=section, authority_note=note,
                      verification_status="PRIVATE_VERIFICATION_STATUS",
                      text=f"{title}\n{chunk_id}. {section}\n{body}\nКонтекст использования: {note}")
        if policy:
            result["index_policy"] = policy
        return result

    def configure(self):
        raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in self.records)
        self.game.rag_config = structured.Data(dict(
            corpus_b64=base64.b64encode(zlib.compress(raw.encode())).decode(),
            embedding_base_url="http://diagnostic.invalid/v1", embedding_api_key="diagnostic-only",
            embedding_model="test-embedding", dimension=3, chroma_path="unused-mocked-path",
            collection_name="test", top_k=2))

    def assert_clean(self, text):
        for technical in ("PRIVATE_AUTHORITY_NOTE", "PRIVATE_VERIFICATION_STATUS", "<source>",
                          "Каталоговый заголовок", "Каталоговый раздел", "[chunk_id]",
                          "ОБЯЗАТЕЛЬНЫЙ КОНТЕКСТ", "Сохраняй формат ответа", "Статус:"):
            self.assertNotIn(technical, text)

    def retrieval(self, fail=False):
        embedding = AsyncMock(side_effect=RuntimeError("offline")) if fail else AsyncMock(
            return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[0.0, 1.0, 0.0])]))

        class EmbeddingClient:
            def __init__(self, **kwargs):
                self.embeddings = SimpleNamespace(create=embedding)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        query = Mock(return_value=dict(ids=[["extra-admin", "holding-id", "shared-id", "held-admin"]],
            metadatas=[[dict(area="methodology"), dict(area="role"), dict(area="shared"),
                        dict(area="legal_reference", index_policy="reference_hold_not_main_index")]]))
        modules = {
            "openai": SimpleNamespace(AsyncOpenAI=EmbeddingClient),
            "chromadb": SimpleNamespace(PersistentClient=lambda **kw: SimpleNamespace(
                get_collection=lambda *args, **kw: SimpleNamespace(query=query))),
            "chromadb.config": SimpleNamespace(Settings=lambda **kw: kw),
        }
        return patch.dict("sys.modules", modules), embedding, query

    async def test_plain_core_preserves_role_restrictions_and_ids_only_in_audit(self):
        context = await self.game._rag_context(structured.NAMES[0], self.players[0]["profile"], [], 1)
        self.assertEqual(context, "SHARED_FACT: смета проекта 270 млн рублей.\n\n"
                                 "ROLE_FACT_admin: соблюдай лимит своей стороны; см. J-04.")
        self.assert_clean(context)
        self.assertNotIn("admin-id", context)
        self.assertNotIn("ROLE_FACT_holding", context)
        self.assertEqual(self.game._rag_audit[-1]["core_chunk_ids"], ["shared-id", "admin-id"])
        self.assertEqual(self.game._rag_audit[-1]["skip_reason"], "no_allowed_sources")

    async def test_extra_retrieval_passes_only_allowed_text_and_audits_ids(self):
        self.records.extend([
            self.record("extra-admin", "methodology", ["admin"], "EXTRA_FACT: проверяемый пример."),
            self.record("held-admin", "legal_reference", ["admin"], "HELD_REFERENCE",
                        "reference_hold_not_main_index"),
        ])
        self.configure()
        mocks, embedding, query = self.retrieval()
        with mocks:
            context = await self.game._rag_context(structured.NAMES[0], self.players[0]["profile"], [], 2)
        self.assertIn("EXTRA_FACT: проверяемый пример.", context)
        self.assertNotIn("HELD_REFERENCE", context)
        self.assertNotIn("ROLE_FACT_holding", context)
        self.assert_clean(context)
        self.assertEqual(self.game._rag_audit[-1]["chunk_ids"], ["extra-admin"])
        self.assertEqual(self.game._rag_audit[-1]["retrieval_status"], "ok")
        embedding.assert_awaited_once()
        self.assertIn({"visible_admin": True}, query.call_args.kwargs["where"]["$and"])

    async def test_failed_extra_search_keeps_core_without_error_text_in_context(self):
        self.records.append(self.record("extra-admin", "methodology", ["admin"], "EXTRA_FACT"))
        self.configure()
        mocks, _, _ = self.retrieval(fail=True)
        with mocks:
            context = await self.game._rag_context(structured.NAMES[0], self.players[0]["profile"], [], 2)
        self.assertIn("SHARED_FACT", context)
        self.assertIn("ROLE_FACT_admin", context)
        self.assertNotIn("offline", context)
        self.assert_clean(context)
        self.assertEqual(self.game._rag_audit[-1]["retrieval_status"], "error")
        self.assertEqual(self.game._rag_audit[-1]["error_type"], "RuntimeError")

    async def test_last_call_and_voting_do_not_request_extra_embeddings(self):
        self.records.append(self.record("extra-admin", "methodology", ["admin"], "EXTRA_FACT"))
        self.configure()
        mocks, embedding, query = self.retrieval()
        with mocks:
            for phase in (2, 3):
                context = await self.game._rag_context(structured.NAMES[0], self.players[0]["profile"],
                                                       [], phase, retrieve=False)
                self.assertIn("ROLE_FACT_admin", context)
                self.assertNotIn("EXTRA_FACT", context)
                self.assert_clean(context)
        embedding.assert_not_awaited()
        query.assert_not_called()

    async def test_disabled_rag_does_not_expose_context(self):
        self.game.use_rag = False
        self.assertEqual(await self.game._rag_context("Администрация", {}, [], 1), "")
        self.assertEqual(self.game._rag_audit, [])

    def check_messages(self):
        for player in self.players:
            role = structured.NS["_rag_role"](player["profile"], player["name"])
            for messages in player["llm"].calls:
                system = "\n".join(m.content for m in messages if m.type == "system")
                human = "\n".join(m.content for m in messages if m.type == "human")
                self.assertIn(player["prompt"], system)
                self.assertIn("RULE_ONLY_IN_SYSTEM", system)
                self.assertIn("Верни один вызов функции PlayerTurn", system)
                self.assertNotIn(player["prompt"], human)
                self.assertNotIn("RULE_ONLY_IN_SYSTEM", human)
                self.assertIn("SHARED_FACT", human)
                self.assertIn("ROLE_FACT_" + role, human)
                self.assertNotIn("ИНСТРУКЦИЯ К LAST CALL", human)
                self.assert_clean(human)

    async def test_all_game_phases_separate_system_instructions_and_role_facts(self):
        result = (await self.game.run_chat()).data
        self.assertEqual(result["status"], "completed")
        self.check_messages()
        self.assertEqual({r["phase"] for r in result["rag_retrievals"]}, {1, 2, 3})
        self.assertTrue(all(r["core_chunk_ids"] for r in result["rag_retrievals"]))

    async def test_last_call_separates_instructions_without_altering_report_renderer(self):
        history = []
        await self.game._run_last_call(self.players, self.state, history,
                                       self.game.rules_prompt, 0, self.conditions, self.cost)
        self.check_messages()
        self.assertTrue(all(h["is_last_call"] for h in history))
        self.assertTrue(all(r["core_chunk_ids"] for r in self.game._rag_audit))
        report = structured.render_dialogue(history, self.state, {}, self.cost, self.conditions)
        self.assertIn("Финальный раунд согласования (Last Call)", report)
        self.assert_clean(report)


if __name__ == "__main__":
    unittest.main()
