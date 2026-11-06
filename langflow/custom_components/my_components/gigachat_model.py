"""GigaChat as a Langflow language model using curl -k equivalent TLS settings."""

import os

import httpx
from langchain_openai import ChatOpenAI
from lfx.custom import Component
from lfx.io import IntInput, Output, SecretStrInput, StrInput


class GigaChatModel(Component):
    display_name = "GigaChat Model"
    description = "GigaChat через OpenAI API с отключённой проверкой TLS-сертификата."
    icon = "bot"
    inputs = [
        SecretStrInput(name="access_token", display_name="Access token", required=False),
        StrInput(name="model_name", display_name="Model ID", value="GigaChat", required=True),
        IntInput(name="timeout_seconds", display_name="Timeout, seconds", value=120, required=True),
    ]
    outputs = [
        Output(name="model_output", display_name="Language Model", method="build_model", types=["LanguageModel"])
    ]

    def build_model(self):
        token = self.access_token or ""
        if hasattr(token, "get_secret_value"):
            token = token.get_secret_value()
        token = str(token).strip() or os.getenv("GIGACHAT_ACCESS_TOKEN", "").strip()
        if not token:
            raise ValueError("Укажите Access token или переменную окружения GIGACHAT_ACCESS_TOKEN")

        timeout = int(self.timeout_seconds)
        return ChatOpenAI(
            base_url="https://api.giga.chat/v1",
            model=str(self.model_name).strip(),
            api_key=token,
            timeout=timeout,
            max_retries=1,
            http_client=httpx.Client(verify=False, timeout=timeout),
            http_async_client=httpx.AsyncClient(verify=False, timeout=timeout),
        )
