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


class ContractJudge(Component):
    display_name = "Contract Judge"
    description = "Анализирует историю переговоров и стратегии участников."
    icon = "gavel"

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
            display_name="Инструкция Судьи",
            info="Промпт/задача для анализа переговоров на русском языке",
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Judge Result",
            method="run",
            types=["Message"],
        )
    ]

    def run(self) -> Message:
        if hasattr(self, "active") and not self.active:
            return Message(text="(Судейский разбор пропущен в быстром режиме разработки)")
        history = self.game_result.data.get("history", [])
        profiles = self.game_result.data.get("players", [])
        instruction = str(getattr(self, "instruction", "") or "").strip()

        if not instruction:
            instruction = (
                "Ты — беспристрастный и опытный судья переговорного процесса.\n\n"
                "Внимательно изучи скрытые профили игроков, их реальные бюджеты и полную историю переговоров.\n\n"
                "Проведи детальный аналитический разбор на русском языке по критериям:\n"
                "- стратегическое поведение и качество переговоров\n"
                "- соответствие скрытым интересам и бюджетам\n"
                "- использование команд SET, SPEAK, PASS и REJECT\n"
                "- утечки информации, блеф и уступки\n"
                "- сила итоговых позиций сторон\n\n"
                "Пиши разбор исключительно на русском языке."
            )

        prompt = (
            f"{instruction}\n\n"
            "ПРОФИЛИ УЧАСТНИКОВ (СКРЫТЫЕ ДАННЫЕ):\n"
            f"{json.dumps(profiles, ensure_ascii=False, indent=2)}\n\n"
            "ПОЛНАЯ ИСТОРИЯ ХОДОВ ПЕРЕГОВОРОВ:\n"
            f"{json.dumps(history, ensure_ascii=False, indent=2)}"
        )

        try:
            llm = get_llm(model=self.model, user_id=self.user_id)
            response = llm.invoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка вызова LLM судьи.\n"
                f"self.model = {self.model!r}\n"
                f"error = {e}"
            ) from e
        raw_content = getattr(response, "content", response)
        result = _extract_text(raw_content)

        return Message(text=result)
