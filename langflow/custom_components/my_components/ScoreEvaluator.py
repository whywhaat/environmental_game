from lfx.custom import Component
from lfx.inputs.inputs import ModelInput, DataInput, MultilineInput
from lfx.io import BoolInput, Output
from lfx.schema.data import Data
from langchain_core.messages import HumanMessage
from lfx.base.models.unified_models import get_llm
import re


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


class ContractScore(Component):
    display_name = "Contract Score"
    description = "Преобразует выводы судьи и прогноз в структурированные баллы (0-100)."
    icon = "badge-check"

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
            is_list=False,
        ),
        MultilineInput(
            name="judge_result",
            display_name="Judge Result",
            required=True,
        ),
        MultilineInput(
            name="prediction",
            display_name="Prediction",
            required=True,
        ),
        MultilineInput(
            name="instruction",
            display_name="Инструкция Скоринга",
            info="Промпт/правила скоринга на русском языке",
            value="",
            required=False,
        ),
    ]

    outputs = [
        Output(
            name="result",
            display_name="Score Result",
            method="run",
            types=["Data"],
        )
    ]

    def run(self) -> Data:
        if hasattr(self, "active") and not self.active:
            return Data(text="(Скоринг пропущен в быстром режиме разработки)", data={"scores": {}, "future_of_project": "(Пропущено)"})
        players = self.game_result.data.get("players", [])
        player_names = [p["player_name"] for p in players]
        instruction = str(getattr(self, "instruction", "") or "").strip()

        if not instruction:
            instruction = (
                "Ты — итоговый модуль скоринга и оценки эффективности переговоров.\n\n"
                "На основе анализа судьи и прогноза проекта выставь каждому участнику оценку от 0 до 100 баллов.\n"
                "Формат ответа:\n"
                "SET <ИМЯ_ИГРОКА> SCORE <0-100>\n"
                "SET <ИМЯ_ИГРОКА> SCORE <0-100>\n"
                "...\n"
                "FUTURE OF PROJECT: <Резюме на русском языке о будущем проекта>"
            )

        prompt = (
            f"{instruction}\n\n"
            f"СПИСОК ИГРОКОВ (используй точные имена):\n{', '.join(player_names)}\n\n"
            f"ВЫВОДЫ СУДЬИ:\n{self.judge_result}\n\n"
            f"ПРОГНОЗ ПРОЕКТА:\n{self.prediction}"
        )

        try:
            llm = get_llm(model=self.model, user_id=self.user_id)
            response = llm.invoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as e:
            raise RuntimeError(
                f"Ошибка вызова LLM скоринга.\n"
                f"self.model = {self.model!r}\n"
                f"error = {e}"
            ) from e

        raw_content = getattr(response, "content", response)
        raw = _extract_text(raw_content)

        scores = {}
        future = None
        names_by_casefold = {}
        for player_name in player_names:
            names_by_casefold.setdefault(player_name.casefold(), []).append(player_name)

        for line in raw.splitlines():
            line = line.strip()

            match = re.match(
                r"^SET\s+(.+?)\s+SCORE\s+(\d+(?:\.\d+)?)$",
                line,
                re.I,
            )

            if match:
                name = match.group(1).strip()
                score = float(match.group(2))

                if name not in player_names:
                    matches = names_by_casefold.get(name.casefold(), [])
                    if not matches:
                        # Try substring match
                        sub = [pn for pn in player_names if name.casefold() in pn.casefold() or pn.casefold() in name.casefold()]
                        if len(sub) == 1:
                            name = sub[0]
                        else:
                            name = None
                    else:
                        name = matches[0]

                if name and 0 <= score <= 100:
                    scores[name] = score
                continue

            match = re.match(
                r"^FUTURE OF PROJECT:\s*(.*)$",
                line,
                re.I,
            )

            if match:
                future = match.group(1).strip()

        if future is None:
            # Fallback: extract last paragraph or summary line
            future = raw.splitlines()[-1] if raw.splitlines() else "Проект находится на стадии реализации."

        result = {
            "scores": scores,
            "future_of_project": future,
            "raw": raw,
        }

        return Data(
            text=raw,
            data=result,
        )
