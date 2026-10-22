"""Langflow-компонент «Модель последствий»: обёртка над прогнозом; в собранном файле модель встроена рядом."""

from __future__ import annotations

from lfx.custom import Component
from lfx.io import DataInput, IntInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

from .catalog import DEFAULT_PROMISED_MONTHS
from .model import forecast


class ConsequencesModel(Component):
    display_name = "Модель последствий"
    description = "К чему придут город N и участники через 10 лет после голосования по протоколу."
    icon = "landmark"

    inputs = [
        DataInput(name="game_result", display_name="Game Result", required=True,
                  info="Выход оркестратора игры."),
        DataInput(name="players", display_name="Players", is_list=True, required=False,
                  info="Профили игроков: из них берутся бюджеты, чтобы видеть кредиты."),
        IntInput(name="promised_months", display_name="Обещанный срок ввода очистных, мес.",
                 value=DEFAULT_PROMISED_MONTHS, info="В игре срок не обсуждается; это допущение модели."),
    ]
    outputs = [
        Output(name="forecast", display_name="Прогноз", method="build_forecast"),
        Output(name="report", display_name="Отчёт", method="build_report"),
        Output(name="herald_prompt", display_name="Промпт «Вестника»", method="build_herald_prompt"),
    ]

    def _forecast(self) -> dict:
        # Прогноз дешёвый и детерминированный: считаем на каждый выход, без кэша между запусками.
        players = [item.data for item in (self.players or [])]
        return forecast(self.game_result.data, players, int(self.promised_months))

    def build_forecast(self) -> Data:
        return Data(data=self._forecast())

    def build_report(self) -> Message:
        return Message(text=self._forecast()["report"])

    def build_herald_prompt(self) -> Message:
        return Message(text=self._forecast()["herald_prompt"])
