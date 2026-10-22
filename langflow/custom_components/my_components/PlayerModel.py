import json

from lfx.custom import Component
from lfx.io import FloatInput, IntInput, ModelInput, MultilineInput, Output, StrInput
from lfx.schema.data import Data


class PlayerAgent(Component):
    display_name = "Player Agent"
    description = "Модель, личный контекст и экономический профиль участника."
    icon = "user-check"
    inputs = [
        ModelInput(name="model", display_name="Language Model", required=True),
        StrInput(name="player_id", display_name="ID игрока", required=True),
        StrInput(name="player_name", display_name="Имя игрока", required=True),
        IntInput(name="budget_millions", display_name="Бюджет, млн ₽", value=0, required=True),
        FloatInput(name="expected_own_share", display_name="Желаемая доля собственного взноса (0–1)", value=0, required=True),
        MultilineInput(name="desired_responsible", display_name="Желаемые ответственные, JSON", required=True,
                       value='{"water_supply":"industrial_holding","independent_monitoring":"industrial_holding","reconstruction":"industrial_holding"}'),
        MultilineInput(name="personal_context", display_name="Личный контекст", required=True, value="Участвуйте в переговорах от своей роли."),
        MultilineInput(name="motivation", display_name="Мотивация", required=True, value="Добейтесь приемлемого договора."),
        MultilineInput(name="personal_history", display_name="Личная история, JSON-массив строк", required=True, value='["Вы участвуете в переговорах о проекте."]'),
    ]
    outputs = [Output(name="player", display_name="Player Agent", method="build_player")]

    def build_player(self) -> Data:
        return Data(data={
            "id": str(self.player_id).strip(), "name": str(self.player_name).strip(),
            "budget": int(self.budget_millions) * 1_000_000,
            "expected_own_share": float(self.expected_own_share),
            "desired_responsible": json.loads(self.desired_responsible),
            "personal_context": str(self.personal_context),
            "motivation": str(self.motivation),
            "personal_history": json.loads(self.personal_history),
            "llm": self.model,
        })
