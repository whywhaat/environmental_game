"""Реальные LFX/ChatOpenAI, HTTP через MockTransport, без сети и квоты GigaChat.

Запускать Python установленного Langflow из корня репозитория:
    python tests/verify_lf_player_turn.py
"""

import asyncio
import base64
import copy
import json
import os
import tempfile
import zlib
from pathlib import Path

import httpx
from jsonschema import Draft202012Validator
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from lfx.custom.eval import eval_custom_component_code
from lfx.schema.data import Data


ROOT = Path(__file__).resolve().parents[1]


def wire_turn(voting=False):
    return dict(speech="Обязательства подтверждаю.",
                own_contribution=dict(action="unchanged" if voting else "set",
                                      amount_millions=0 if voting else 54),
                own_conditions=[], pledge_demands=[], condition_demands=[],
                responses=[], withdrawn_demands=[], addressed_to="everyone",
                vote="for" if voting else "none")


async def verify():
    component = eval_custom_component_code(
        (ROOT / "langflow/custom_components/my_components/dialog.py").read_text())
    requests = []
    scripted = []
    validate_reply = False
    require_context = False

    async def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        system = "\n".join(m["content"] for m in body["messages"] if m["role"] == "system")
        human = "\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
        assert "ROLE_SYSTEM_MARKER" in system and "RULE_SYSTEM_MARKER" in system
        assert "ROLE_SYSTEM_MARKER" not in human and "RULE_SYSTEM_MARKER" not in human
        if require_context:
            assert "SCENARIO_FACT" in human
            assert "ROLE_LIMIT" in human
            for marker in ("SOURCE_ONLY_", "PRIVATE_METADATA", "<source>", "ОБЯЗАТЕЛЬНЫЙ КОНТЕКСТ"):
                assert marker not in human, marker
        function = body["tools"][0]["function"]
        assert function["name"] == "PlayerTurn"
        parameters = function["parameters"]
        Draft202012Validator.check_schema(parameters)
        serialized = json.dumps(parameters)
        for unsupported in ('"anyOf"', '"oneOf"', '"allOf"', '"type": "null"'):
            assert unsupported not in serialized, unsupported
        voting = parameters["properties"]["vote"]["enum"] != ["none"]
        payload = scripted.pop(0) if scripted else wire_turn(voting)
        if validate_reply:
            Draft202012Validator(parameters).validate(payload)
        return httpx.Response(200, json={
            "id": "transport-check", "object": "chat.completion", "created": 0,
            "model": "diagnostic-only", "choices": [{"index": 0, "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-check", "type": "function", "function": {
                        "name": "PlayerTurn", "arguments": json.dumps(payload, ensure_ascii=False)}}]}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        model = ChatOpenAI(base_url="http://diagnostic.invalid/v1", api_key="diagnostic-only",
                           model="diagnostic-only", max_retries=0, http_async_client=client)
        def game():
            instance = component()
            ids = ["city_administration", "industrial_holding", "regional_government",
                   "city_residents", "environmental_ngo"]
            names = ["Администрация", "Промкомплекс", "Регион", "Жители", "Эко-НКО"]
            instance.players = [Data(data=dict(id=pid, name=name, budget=200_000_000,
                                                prompt="ROLE_SYSTEM_MARKER: проверка контракта.", llm=model))
                                for pid, name in zip(ids, names)]
            records = [dict(chunk_id="SOURCE_ONLY_shared", area="shared", audience=["admin", "holding", "region", "residents", "ngo"],
                            text="SCENARIO_FACT: стоимость проекта 270 млн рублей.", verification_status="PRIVATE_METADATA")]
            for role in ("admin", "holding", "region", "residents", "ngo"):
                records.append(dict(chunk_id="SOURCE_ONLY_" + role, area="role", audience=[role],
                                    text="ROLE_LIMIT: соблюдай лимит своей стороны.", authority_note="PRIVATE_METADATA"))
            raw = "\n".join(json.dumps(record, ensure_ascii=False) for record in records).encode()
            instance.rag_config = Data(data={"corpus_b64": base64.b64encode(zlib.compress(raw)).decode()})
            instance.rules_prompt = "RULE_SYSTEM_MARKER: регламент проверки."
            instance.turns = 5
            instance.request_attempts = 1
            instance.request_timeout_seconds = 10
            instance.game_timeout_seconds = 90
            instance._debug_calls = []
            instance._run_store = None
            return instance

        instance = game()
        players, _, state, settings = instance._init_session()
        bad = wire_turn()
        bad["own_contribution"]["amount_millions"] = True
        scripted.extend([bad, wire_turn()])
        answer, action = await instance._structured_turn(players[0], "Торг", 1,
            [HumanMessage(content="Проверка исправления")], state, settings["levers"], settings["total_cost"])
        assert answer is not None
        assert state["players"][players[0]["name"]]["contribution"] == 54_000_000
        repair = requests[-1]["messages"][-1]["content"]
        assert '"loc": ["own_contribution", "amount_millions"]' in repair
        assert "response_metadata" not in repair
        assert len(requests) == 2
        print("REAL_STACK: tool arguments -> ошибки по полям -> исправление -> принятый ход", flush=True)

        legacy = dict(speech="Общая позиция.", own_contribution_millions="null", own_levers={},
                      demands=[], responses=[], withdrawn_demands=[], addressed_to="все", vote="null")
        scripted.append(legacy)
        before = copy.deepcopy(state)
        answer, _ = await instance._structured_turn(players[0], "Торг", 2,
            [HumanMessage(content="Проверка нормализации")], state, settings["levers"], settings["total_cost"])
        assert answer is not None and state == before
        assert len(instance._debug_calls[-1]["normalization"]) == 3
        print("REAL_STACK: артефакты null/все нормализованы до внутренней проверки", flush=True)

        validate_reply = True
        residents = wire_turn()
        residents["own_contribution"] = dict(action="unchanged", amount_millions=0)
        residents["condition_demands"] = [
            dict(target="regional_government", lever_id="guilt", value_json="holding",
                 text="Признать вину холдинга", counter_to=""),
            dict(target="industrial_holding", lever_id="plant_mode", value_json="70%",
                 text="Снизить мощность", counter_to="")]
        scripted.append(residents)
        answer, _ = await instance._structured_turn(players[3], "Торг", 3,
            [HumanMessage(content="Проверка реальных строковых значений")], state, settings["levers"], settings["total_cost"])
        assert answer is not None
        assert answer.demands[0].value == "holding" and answer.demands[1].value == "70%"
        print("REAL_STACK: holding и 70% приняты без дополнительного JSON-кодирования", flush=True)

        previous = os.environ.get("NEGOTIATION_GAMES_DIR")
        require_context = True
        try:
            with tempfile.TemporaryDirectory(prefix="ecolog-player-turn-") as directory:
                os.environ["NEGOTIATION_GAMES_DIR"] = directory
                instance = game()
                result = (await asyncio.wait_for(instance.run_chat(), 90)).data
                assert result["status"] == "completed", result.get("error")
                assert result["consensus_reached"]
                assert all(h["structured_response"] for h in result["history"])
                assert all(r["core_chunk_ids"] for r in result["rag_retrievals"])
                assert Path(result["files"]["result.json"]).exists()
                print("REAL_STACK: полный граф оркестратора завершён; все ходы приняты, отчёт сохранён", flush=True)
                print("REAL_STACK: роль/регламент в system, факты без метаданных в user, ID источников в аудите", flush=True)
        finally:
            if previous is None:
                os.environ.pop("NEGOTIATION_GAMES_DIR", None)
            else:
                os.environ["NEGOTIATION_GAMES_DIR"] = previous


if __name__ == "__main__":
    asyncio.run(verify())
