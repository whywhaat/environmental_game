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
from lfx.io import FloatInput, ModelInput, MultilineInput, Output, SecretStrInput, SliderInput
from lfx.schema.data import Data


class PlayerAgent(LCModelComponent):
    model_provider_policy_mode = "delegate"
    display_name = "Player Agent"
    description = "Defines a player role, economic profile, and language model."
    icon = "user-check"
    category = "models"

    inputs = [
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Select your model provider",
            real_time_refresh=True,
            required=True,
        ),
        StrInput(
            name="model_name",
            display_name="Model Name Override",
            info="Optional model name override",
            advanced=True,
            load_from_db=False,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Overrides global provider settings.",
            required=False,
            show=True,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        StrInput(
            name="player_name",
            display_name="Player name",
            info="Unique player name",
            value="",
            required=True,
        ),
        MultilineInput(
            name="prompt",
            display_name="Prompt",
            info="System prompt/personality for this player.",
            value="You are a strategic negotiator.",
            required=True,
        ),
        FloatInput(
            name="money",
            display_name="Money",
            info="Initial amount of money physically available to the player.",
            value=100000,
            required=True,
        ),
        FloatInput(
            name="min_share",
            display_name="Minimum Share (%)",
            info="Below this share the player considers the deal unfavorable.",
            value=0,
            required=True,
        ),
        FloatInput(
            name="comfort_share",
            display_name="Comfort Share (%)",
            info="Share from which accepting the deal becomes comfortable.",
            value=0,
            required=True,
        ),
        FloatInput(
            name="comfort_money",
            display_name="Comfort Money",
            info="Preferred amount the player would like to invest.",
            value=0,
            required=True,
        ),
        FloatInput(
            name="max_money",
            display_name="Maximum Money",
            info="Maximum amount the player is willing to invest.",
            value=0,
            required=True,
        ),
        MultilineInput(
            name="intent",
            display_name="Intent",
            info="Strategic preference and desired direction of optimization.",
            value="",
            required=True,
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
            user_id=self.user_id,
            get_options=get_language_model_options,
        )
        return get_llm(
            model=model,
            user_id=self.user_id,
            api_key=getattr(self, "api_key", None),
            temperature=getattr(self, "temperature", 0.1),
        )

    def build_player(self) -> Data:
        money = float(self.money)
        min_share = float(self.min_share)
        comfort_share = float(self.comfort_share)
        comfort_money = float(self.comfort_money)
        max_money = float(self.max_money)

        if money < 0:
            raise ValueError("Money cannot be negative.")

        if not 0 <= min_share <= 100:
            raise ValueError("Minimum Share must be between 0 and 100.")

        if not 0 <= comfort_share <= 100:
            raise ValueError("Comfort Share must be between 0 and 100.")

        if comfort_share < min_share:
            raise ValueError(
                "Comfort Share cannot be less than Minimum Share."
            )

        if comfort_money < 0:
            raise ValueError("Comfort Money cannot be negative.")

        if max_money < 0:
            raise ValueError("Maximum Money cannot be negative.")

        if comfort_money > max_money:
            raise ValueError(
                "Comfort Money cannot be greater than Maximum Money."
            )

        if max_money > money:
            raise ValueError(
                "Maximum Money cannot be greater than Money."
            )

        if not str(self.player_name).strip():
            raise ValueError("Player name cannot be empty.")

        if not str(self.prompt).strip():
            raise ValueError("Prompt cannot be empty.")

        llm = self.build_llm()

        return Data(
            data={
                "player_name": str(self.player_name).strip(),
                "llm": llm,
                "prompt": str(self.prompt).strip(),
                "money": money,
                "attrs": {
                    "min_share": min_share,
                    "comfort_share": comfort_share,
                    "comfort_money": comfort_money,
                    "max_money": max_money,
                },
                "intent": str(self.intent).strip(),
            }
        )

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        """Dynamically update build config with user-filtered model options."""
        return handle_model_input_update(self, build_config, field_value, field_name)
