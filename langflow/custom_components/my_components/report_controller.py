import json

from lfx.custom import Component
from lfx.io import BoolInput, DataInput, Output
from lfx.schema.message import Message


class ReportController(Component):
    display_name = "Report Controller"
    description = "Читаемый итог переговоров."
    icon = "file-text"
    inputs = [
        DataInput(name="game_result", display_name="Game Result", required=True),
        BoolInput(name="debug_mode", display_name="Debug mode", value=False),
    ]
    outputs = [Output(name="result", display_name="Итоговый вывод", method="generate_report")]

    def generate_report(self) -> Message:
        payload = self.game_result.data
        text = payload["report"]
        if getattr(self, "debug_mode", False):
            details = json.dumps(payload.get("debug", {}), ensure_ascii=False, indent=2)
            fence = "```"
            while fence in details:
                fence += "`"
            text += f"\n\n## Debug mode\n\n{fence}json\n{details}\n{fence}"
        return Message(text=text)
