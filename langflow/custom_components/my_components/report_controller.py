from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema.message import Message


class ReportController(Component):
    display_name = "Report Controller"
    description = "Читаемый итог переговоров и пути к сохранённым файлам."
    icon = "file-text"
    inputs = [DataInput(name="game_result", display_name="Game Result", required=True)]
    outputs = [Output(name="result", display_name="Итоговый вывод", method="generate_report")]

    def generate_report(self) -> Message:
        payload = self.game_result.data
        files = payload["files"]
        paths = "\n".join(f"- `{name}`: `{path}`" for name, path in files.items())
        return Message(text=payload["report"] + "\n\n## Файлы запуска\n\n" + paths)
