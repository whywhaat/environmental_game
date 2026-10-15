from lfx.custom import Component
from lfx.inputs.inputs import DropdownInput, DataInput, MultilineInput
from lfx.io import Output
from lfx.schema.message import Message


class ReportController(Component):
    display_name = "Report Controller"
    description = "Переключатель режимов вывода: чистый Протокол слушаний, аналитика или полный отчет."
    icon = "sliders-horizontal"

    inputs = [
        DropdownInput(
            name="mode",
            display_name="Режим вывода",
            options=[
                "Только диалог и контракт (Быстро)",
                "Аналитический отчет (Судья + Прогноз + Оценки)",
                "Все вместе (Диалог + Контракт + Аналитика)",
            ],
            value="Только диалог и контракт (Быстро)",
            info="Выберите формат отчета для отображения в чате.",
        ),
        DataInput(
            name="game_result",
            display_name="Game Result",
            info="Результат симуляции с протоколом и голосованием",
            required=True,
        ),
        MultilineInput(
            name="judge_result",
            display_name="Judge Result",
            info="Разбор от судьи",
            required=False,
        ),
        MultilineInput(
            name="prediction",
            display_name="Prediction",
            info="Прогноз проекта",
            required=False,
        ),
        DataInput(
            name="score_result",
            display_name="Score Result",
            info="Баллы участников",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Итоговый вывод",
            method="generate_report",
            types=["Message"],
        )
    ]

    def _render_analytics(self) -> str:
        parts = ["# ⚖️ Аналитический отчет по итогам слушаний\n"]

        judge = str(getattr(self, "judge_result", "") or "").strip()
        if judge:
            parts.append(f"### 1. 🔍 Разбор Судьи (Тактика и стратегия сторон):\n{judge}\n")

        pred = str(getattr(self, "prediction", "") or "").strip()
        if pred:
            parts.append(f"### 2. 📈 Инвестиционный и экологический прогноз:\n{pred}\n")

        score_data = getattr(self, "score_result", None)
        scores_dict = {}
        future = ""
        if score_data and hasattr(score_data, "data") and isinstance(score_data.data, dict):
            scores_dict = score_data.data.get("scores", {})
            future = score_data.data.get("future_of_project", "")
        elif score_data and hasattr(score_data, "text"):
            future = str(score_data.text)

        if scores_dict:
            parts.append("### 3. 🏆 Оценки участников (Скоринг):\n| Участник | Баллы (0-100) |\n| :--- | :--- |")
            for p_name, sc in scores_dict.items():
                parts.append(f"| **{p_name}** | **{sc:g} / 100** |")

        if future:
            parts.append(f"\n### 🔮 Будущее проекта:\n{future}\n")

        return "\n".join(parts)

    def generate_report(self) -> Message:
        protocol_text = ""
        if hasattr(self.game_result, "text") and self.game_result.text:
            protocol_text = str(self.game_result.text).strip()

        mode = str(self.mode)

        if "Только диалог" in mode:
            return Message(text=protocol_text)
        elif "Аналитический отчет" in mode:
            analytics = self._render_analytics()
            return Message(text=analytics)
        else:
            analytics = self._render_analytics()
            full_report = f"{protocol_text}\n\n---\n\n{analytics}"
            return Message(text=full_report)
