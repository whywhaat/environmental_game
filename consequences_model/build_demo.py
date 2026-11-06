"""Самостоятельный стенд последствий: сохранённый проверочный итог → обычный компонент модели."""

from __future__ import annotations

import copy
import inspect
import json
import uuid
from pathlib import Path

from .build_flow import CHAT, FLOW_PATH, NODE, build_edge, component_template, write_flow

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/consequences"
DEMO_FLOW_PATH = FLOW_PATH.parent / "Consequences_demo.json"
DEMO_FLOW_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "environmental_game/consequences-demo/v1"))
INPUT = "ChatInput-consequences-demo"
SELECTOR = "ConsequencesExample-main"


def read_examples() -> list[dict]:
    """Примеры в формате Data.data; порядок в manifest задаёт номера в чате."""
    manifest = json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8"))
    return [{**case, "game_result": json.loads((EXAMPLES / f"{case['id']}.json").read_text(encoding="utf-8"))}
            for case in manifest]


def choose_example(query: str, cases: list[dict]) -> dict:
    """Номер, id или название → независимая копия проверочного итога; без случайного выбора."""
    choice = " ".join((query or "").split()).casefold() or "1"
    for number, case in enumerate(cases, 1):
        if choice in (str(number), case["id"].casefold(), case["name"].casefold()):
            return copy.deepcopy(case)
    options = "; ".join(f"{number}. {case['name']}" for number, case in enumerate(cases, 1))
    raise ValueError(f"Неизвестный пример «{query}». Напишите номер, название или id исхода: {options}")


def render_selector() -> str:
    """Встраивает проверочные JSON и тот же выбор примера в автономный компонент."""
    data = repr(json.dumps(read_examples(), ensure_ascii=False))
    return ('''"""Проверочный итог для стенда последствий. Собрано build_demo; не сыгранная партия."""
import copy
import json

from lfx.custom import Component
from lfx.io import MessageTextInput, Output
from lfx.schema.data import Data

''' + f"CASES = json.loads({data})\n\n\n" + inspect.getsource(choose_example) + '''

class ConsequencesExample(Component):
    display_name = "Проверочный итог"
    description = "Выбор одного из 10 синтетических итогов игры, без LLM."
    icon = "ListOrdered"
    inputs = [MessageTextInput(name="input_value", display_name="Номер или название исхода", value="1",
                              info="Номер 1–10, название или id исхода. Например: 1 или Открытая река.")]
    outputs = [Output(name="game_result", display_name="Итог для проверки", method="build_result")]

    def build_result(self) -> Data:
        case = choose_example(self.input_value, CASES)
        self.status = "Проверочный пример: " + case["name"]
        return Data(data=case["game_result"])
''')


def build_demo(source: dict, selector: dict, chat_input: dict) -> dict:
    """Собирает только ChatInput → проверочный итог → модель → ChatOutput."""
    original = {node["id"]: node for node in source["data"]["nodes"]}
    model, chat = copy.deepcopy(original[NODE]), copy.deepcopy(original[CHAT])
    version = model["data"]["node"]["lf_version"]

    def node(node_id: str, kind: str, info: dict) -> dict:
        info = copy.deepcopy(info)
        info["lf_version"] = version
        info["template"]["_frontend_node_flow_id"] = {"value": node_id, "name": "_frontend_node_flow_id", "type": "str"}
        return {"id": node_id, "type": "genericNode", "selected": False,
                "data": {"id": node_id, "type": kind, "node": info, "showNode": False,
                         "selected_output": info["outputs"][0]["name"]}}

    input_node = node(INPUT, "ChatInput", chat_input)
    input_node["data"]["node"]["template"]["input_value"].update(
        value="1", placeholder="Номер 1–10 или название исхода")
    selector_node = node(SELECTOR, "ConsequencesExample", selector)
    chat["data"]["node"]["template"]["sender_name"]["value"] = "Вестник · проверочный пример"
    nodes = [input_node, selector_node, model, chat]
    by_id = {item["id"]: item for item in nodes}
    for index, item in enumerate(nodes):
        item["position"] = {"x": index * 350, "y": 0}
        item["measured"] = {"width": 250, "height": 100}

    options = "; ".join(f"{number}. {case['name']}" for number, case in enumerate(read_examples(), 1))
    return {"id": DEMO_FLOW_ID, "name": "Модель последствий · стенд 10 исходов",
            "description": "Проверочные примеры, не сыгранные партии. Без ядра игры и LLM. "
                           "В Playground отправьте номер, название или id исхода. " + options,
            "data": {"nodes": nodes, "edges": [build_edge(by_id, INPUT, "message", SELECTOR, "input_value"),
                                                build_edge(by_id, SELECTOR, "game_result", NODE, "game_result"),
                                                build_edge(by_id, NODE, "newspaper", CHAT, "input_value")],
                     "viewport": {"x": 40, "y": 200, "zoom": 0.8}}}


def main() -> None:
    from lfx.components.input_output.chat import ChatInput

    source = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    input_code = Path(inspect.getfile(ChatInput)).read_text(encoding="utf-8")
    write_flow(build_demo(source, component_template(render_selector()), component_template(input_code)), DEMO_FLOW_PATH)
    print(f"Собрано: {DEMO_FLOW_PATH}")


if __name__ == "__main__":
    main()
