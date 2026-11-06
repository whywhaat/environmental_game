import os

from langchain_openai import ChatOpenAI
from lfx.custom import Component
from lfx.io import IntInput, Output, StrInput


class OpenAICompatibleModel(Component):
    display_name = "OpenAI Compatible Model"
    description = "Локальная или внешняя OpenAI-совместимая модель."
    icon = "bot"
    inputs = [
        StrInput(name="base_url", display_name="Base URL", value="http://host.docker.internal:9090/v1", required=True),
        StrInput(name="model_name", display_name="Model ID", value="qwen3.8-9b-q4", required=True),
        IntInput(name="timeout_seconds", display_name="Timeout, seconds", value=360, required=True),
        IntInput(name="max_tokens", display_name="Maximum response tokens", value=512, required=True),
    ]
    outputs = [Output(name="model_output", display_name="Language Model", method="build_model", types=["LanguageModel"])]

    def build_model(self):
        return ChatOpenAI(
            base_url=str(self.base_url).rstrip("/"), model=str(self.model_name),
            api_key=os.getenv("OPENAI_API_KEY") or "localai",
            timeout=int(self.timeout_seconds), max_retries=1, max_tokens=int(self.max_tokens),
        )
