"""Langflow-компонент «Модель последствий»: обёртка над прогнозом; в собранном файле модель встроена рядом."""

from __future__ import annotations

from typing import Any

from lfx.custom import Component
from lfx.io import DataInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

from .model import forecast


class ConsequencesModel(Component):
    display_name = "Модель последствий"
    description = "К чему придут город N и участники через 10 лет после голосования по протоколу."
    icon = "landmark"

    inputs = [
        DataInput(name="game_result", display_name="Game Result", required=True,
                  info="Выход оркестратора игры: взносы, условия протокола, голоса и профили игроков."),
    ]
    outputs = [
        Output(name="forecast", display_name="Прогноз", method="build_forecast"),
        Output(name="report", display_name="Отчёт", method="build_report"),
        Output(name="herald_prompt", display_name="Промпт «Вестника»", method="build_herald_prompt"),
    ]

    def _forecast(self) -> dict[str, Any]:
        # Прогноз дешёвый и детерминированный: считаем на каждый выход, без кэша между запусками.
        return forecast(self.game_result.data)

    def build_forecast(self) -> Data:
        return Data(data=self._forecast())

    def build_report(self) -> Message:
        return Message(text=self._forecast()["report"])

    def build_herald_prompt(self) -> Message:
        return Message(text=self._forecast()["herald_prompt"])
