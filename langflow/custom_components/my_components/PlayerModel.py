import json
import re
from typing import Any

from lfx.base.models.model import LCModelComponent
from lfx.base.models.unified_models import (
    get_language_model_options,
    get_llm,
    handle_model_input_update,
)
from lfx.components.models_and_agents.model_selection import apply_model_overrides
from lfx.field_typing.constants import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import StrInput
from lfx.io import FloatInput, IntInput, ModelInput, MultilineInput, Output, SecretStrInput, SliderInput
from lfx.schema.data import Data


class PlayerAgent(LCModelComponent):
    model_provider_policy_mode = "delegate"
    display_name = "Player Agent"
    description = "Единая нода участника: языковая модель, бюджет и стратегия переговоров."
    icon = "user-check"
    category = "models"

    inputs = [
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Выберите провайдера модели (GigaChat, OpenAI, Ollama, Anthropic и др.)",
            real_time_refresh=True,
            required=True,
        ),
        StrInput(
            name="player_name",
            display_name="Имя игрока",
            info="Отображаемое имя участника (например, Администрация города N, Промышленный холдинг)",
            required=True,
            value="",
        ),
        IntInput(
            name="budget_millions",
            display_name="Бюджет, млн ₽",
            value=0,
            info="Физически доступный бюджет участника в миллионах рублей",
            required=True,
        ),
        MultilineInput(
            name="prompt",
            display_name="Стратегия и характер агента",
            info="Инструкция для агента: характер, тактика переговоров, аргументы, красные линии и условия компромисса",
            required=True,
            multiline=True,
            value="Вы представляете интересы своей стороны. Добивайтесь заключения договора, отстаивая свои интересы и не допуская превышения бюджета.",
        ),
        StrInput(
            name="player_id",
            display_name="ID игрока",
            info="Идентификатор роли (city_administration, industrial_holding, regional_government, city_residents, environmental_ngo)",
            required=False,
            advanced=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name Override",
            info="Опциональное переопределение имени модели (например, GigaChat-Pro, qwen3.8-9b-q4)",
            advanced=True,
            load_from_db=False,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API-ключ провайдера (если не задан в переменных окружения)",
            required=False,
            show=True,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Степень креативности модели (0.0 — строгая логика, 1.0 — случайность)",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        FloatInput(
            name="expected_own_share",
            display_name="Желаемая доля взноса (0–1)",
            value=0.0,
            info="Целевая доля участия от 0 до 1 (по умолчанию 0 — минимизировать расходы)",
            advanced=True,
            required=False,
        ),
        MultilineInput(
            name="desired_responsible",
            display_name="Желаемые ответственные (JSON)",
            required=False,
            value="",
            info="Опциональный JSON со списком желаемых ответственных (по умолчанию определяются ролью)",
            advanced=True,
        ),
        MultilineInput(
            name="personal_context",
            display_name="Базовый контекст роли",
            required=False,
            value="",
            info="Официальный контекст и статус участника в сценарии",
            advanced=True,
        ),
        MultilineInput(
            name="motivation",
            display_name="Мотивация",
            required=False,
            value="",
            info="Базовая стратегическая цель участника",
            advanced=True,
        ),
        MultilineInput(
            name="personal_history",
            display_name="Личная история (JSON)",
            required=False,
            value="",
            info="Опциональные факты предыстории (основные факты подтягиваются из RAG)",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            name="player",
            display_name="Player Agent",
            method="build_player",
            types=["Data", "JSON"],
        )
    ]

    def build_llm(self) -> LanguageModel:
        model = apply_model_overrides(
            self.model,
            model_name=getattr(self, "model_name", None),
            provider=getattr(self, "provider", None),
            user_id=getattr(self, "user_id", None),
            get_options=get_language_model_options,
        )
        return get_llm(
            model=model,
            user_id=getattr(self, "user_id", None),
            api_key=getattr(self, "api_key", None),
            temperature=getattr(self, "temperature", 0.1),
        )

    def build_player(self) -> Data:
        if hasattr(getattr(self, "model", None), "invoke"):
            llm = self.model
        else:
            llm = self.build_llm()

        name = str(getattr(self, "player_name", "") or "").strip()
        player_id = str(getattr(self, "player_id", "") or "").strip()
        if not player_id:
            lower_name = name.lower()
            slug_map = {
                "администраци": "city_administration",
                "промкомплекс": "industrial_holding",
                "холдинг": "industrial_holding",
                "регион": "regional_government",
                "жител": "city_residents",
                "эко-нко": "environmental_ngo",
                "нко": "environmental_ngo",
            }
            player_id = next((v for k, v in slug_map.items() if k in lower_name), re.sub(r"[^a-z0-9_]+", "_", lower_name).strip("_") or "player")

        desired_raw = getattr(self, "desired_responsible", "") or ""
        if isinstance(desired_raw, str):
            desired_raw = desired_raw.strip()
            desired = json.loads(desired_raw) if desired_raw else {}
        else:
            desired = desired_raw or {}

        history_raw = getattr(self, "personal_history", "") or ""
        if isinstance(history_raw, str):
            history_raw = history_raw.strip()
            if history_raw:
                try:
                    history = json.loads(history_raw)
                except Exception:
                    history = [line.strip() for line in history_raw.splitlines() if line.strip()]
            else:
                history = []
        else:
            history = history_raw or []

        prompt_text = str(getattr(self, "prompt", "") or "").strip()
        context_text = str(getattr(self, "personal_context", "") or "").strip()
        if prompt_text and not context_text:
            context_text = prompt_text
        elif prompt_text and context_text and prompt_text != context_text:
            context_text = f"{context_text}\n\n### Стратегия и инструкции:\n{prompt_text}"
        elif not context_text:
            context_text = f"Вы представляете {name or player_id} в переговорах."

        motivation_text = str(getattr(self, "motivation", "") or "").strip()
        if not motivation_text:
            motivation_text = f"Защитить интересы стороны {name or player_id} и добиться заключения сбалансированного договора."

        share_val = getattr(self, "expected_own_share", 0.0)
        try:
            share = float(share_val if share_val not in (None, "") else 0.0)
        except (ValueError, TypeError):
            share = 0.0

        budget_m = getattr(self, "budget_millions", 0)
        try:
            budget = int(budget_m) * 1_000_000
        except (ValueError, TypeError):
            budget = 0

        return Data(
            data={
                "id": player_id,
                "name": name or player_id,
                "budget": budget,
                "expected_own_share": share,
                "desired_responsible": desired,
                "personal_context": context_text,
                "motivation": motivation_text,
                "personal_history": history,
                "prompt": prompt_text,
                "llm": llm,
            }
        )

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        return handle_model_input_update(self, build_config, field_value, field_name)
