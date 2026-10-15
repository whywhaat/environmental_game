from lfx.custom import Component
from lfx.inputs.inputs import ModelInput, DataInput
from lfx.io import BoolInput, MultilineInput, Output
from lfx.schema.message import Message
from langchain_core.messages import HumanMessage
from lfx.base.models.unified_models import get_llm
import json


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content).strip()


class ContractPredict(Component):
    display_name = "Contract Predict"
    description = "Прогнозирует будущее развитие и экономические перспективы проекта."
    icon = "trending-up"

    inputs = [
        BoolInput(
            name="active",
            display_name="Активировать модуль",
            info="Выключите для мгновенного пропуска на этапе быстрой разработки",
            value=True,
        ),
        ModelInput(
            name="model",
            display_name="Judge Model",
            required=True,
            model_type="language",
            is_list=False,
        ),
        DataInput(
            name="game_result",
            display_name="Game Result",
            required=True,
        ),
        MultilineInput(
            name="instruction",
            display_name="Инструкция Прогнозиста",
            info="Промпт/задача прогнозирования на русском языке",
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Prediction",
            method="run",
            types=["Message"],
        )
    ]

    def run(self) -> Message:
        if hasattr(self, "active") and not self.active:
            return Message(text="(Экономический прогноз пропущен в быстром режиме разработки)")
        history = self.game_result.data.get("history", [])

        if not history:
            raise ValueError("Game Result contains no history.")

        state = history[-1]["state"]
        instruction = str(getattr(self, "instruction", "") or "").strip()

        if not instruction:
            instruction = (
                "Ты — ведущий инвестиционный и проектный аналитик-прогнозист.\n\n"
                "Проанализируй текущее итоговое состояние проекта и составь подробный прогноз его будущего на русском языке.\n\n"
                "Оцени: взносы, распределение долей, баланс стимулов, инвестиционную привлекательность и устойчивость проекта.\n"
                "Пиши исключительно на русском языке."
            )

        prompt = (
            f"{instruction}\n\n"
            "ФИНАЛЬНОЕ СОСТОЯНИЕ ПРОЕКТА (ВЗНОСЫ И ДОЛИ):\n"
            f"{json.dumps(state, ensure_ascii=False, indent=2)}"
        )

        try:
            llm = get_llm(model=self.model, user_id=self.user_id)
            response = llm.invoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка вызова LLM прогнозиста.\n"
                f"self.model = {self.model!r}\n"
                f"error = {e}"
            ) from e

        raw_content = getattr(response, "content", response)
        result = _extract_text(raw_content)

        return Message(text=result)
