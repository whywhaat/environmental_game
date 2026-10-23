import os
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from lfx.custom import Component
from lfx.io import DataInput, ModelInput, Output
from lfx.schema.data import Data

from negotiation_mvp.langflow_bridge import run_langflow_game


def _as_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content).strip()


def _invoke(model, messages):
    types = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    response = model.invoke([types[item["role"]](content=item["content"]) for item in messages])
    return _as_text(getattr(response, "content", response))


class TwoLLMContractGame(Component):
    display_name = "Two LLM Contract Game"
    description = "Последовательные переговоры через общее Python-ядро."
    icon = "messages-square"
    inputs = [
        DataInput(name="settings", display_name="Game Settings", required=True),
        DataInput(name="players", display_name="Players", is_list=True, required=True),
        DataInput(name="rag_config", display_name="RAG Config (optional)", required=False),
        ModelInput(name="converter_model", display_name="SET Converter Model (optional)", required=False),
    ]
    outputs = [Output(name="result", display_name="Game Result", method="run_chat")]

    def run_chat(self) -> Data:
        settings = self.settings.data
        players = [item.data for item in self.players]
        invokers = {player["id"]: (lambda messages, model=player["llm"]: _invoke(model, messages)) for player in players}
        converter_llm = getattr(self, "converter_model", None) or players[0]["llm"]
        invokers["__converter__"] = lambda messages: _invoke(converter_llm, messages)
        rag_data = getattr(self.rag_config, "data", None) if getattr(self, "rag_config", None) else None
        payload = run_langflow_game(settings, players, invokers, Path(os.getenv("NEGOTIATION_GAMES_DIR", "/app/games")), rag_config=rag_data)
        return Data(data={"result": payload["result"], "files": payload["files"], "report": payload["report"]})
