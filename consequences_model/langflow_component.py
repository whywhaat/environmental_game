"""Langflow-компонент «Модель последствий»: обёртка над прогнозом; в собранном файле модель встроена рядом."""

from __future__ import annotations

import base64
from pathlib import Path, PurePosixPath
from typing import Any

from lfx.custom import Component
from lfx.io import DataInput, FileInput, Output
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
        # Одиночные поля поддерживаются и AST-загрузчиком custom-компонентов Langflow.
        FileInput(name="image_open_river", display_name="Картинка: Открытая река",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_nedostroy", display_name="Картинка: Недострой",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_without_residents", display_name="Картинка: Решили без жителей",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_dolgostroy", display_name="Картинка: Долгострой",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_holding_paid", display_name="Картинка: Холдинг заплатил за всех",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_city_paid", display_name="Картинка: Город заплатил за всех",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_closed_data", display_name="Картинка: Чистая вода, закрытые данные",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_compromise", display_name="Картинка: Компромисс",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_each_for_self", display_name="Картинка: Каждый сам за себя",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
        FileInput(name="image_stagnation", display_name="Картинка: Застой",
                  file_types=["png", "jpg", "jpeg", "webp"], required=False, temp_file=True),
    ]
    outputs = [
        Output(name="forecast", display_name="Прогноз", method="build_forecast"),
        Output(name="report", display_name="Отчёт", method="build_report"),
        Output(name="newspaper", display_name="Газета", method="build_newspaper"),
        Output(name="herald_prompt", display_name="Промпт «Вестника»", method="build_herald_prompt"),
    ]

    def _forecast(self) -> dict[str, Any]:
        # Прогноз дешёвый и детерминированный: считаем на каждый выход, без кэша между запусками.
        return forecast(self.game_result.data)

    def build_forecast(self) -> Data:
        return Data(data=self._forecast())

    def build_report(self) -> Message:
        return Message(text=self._forecast()["report"])

    def build_newspaper(self) -> Message:
        result = self._forecast()
        if result.get("status") == "skipped":
            self.status = "Игра не завершена"
            return Message(text=result["herald_text"], files=[], content_blocks=[])
        image = getattr(self, "image_" + result["outcome"]["id"], None)
        files = []
        blocks = []
        if image:
            # FileInput проверяет доступ и разрешает storage key в полный путь.
            # Браузеру нужен прежний ключ «flow_id/filename», а не путь на сервере.
            path = PurePosixPath(str(image).replace("\\", "/"))
            files = [str(PurePosixPath(*path.parts[-2:]))]
            mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            if path.suffix.lower() not in mime_types:
                raise ValueError("Иллюстрация должна быть PNG, JPEG или WebP")
            # AI-сообщения Playground рисуют rich content, а не Message.files.
            # Группа сохраняет изображение и при сериализации в старый v1 API.
            # Base64 не зависит от URL сервера и cookie браузера; пиксели не меняются.
            blocks = [{"type": "group", "title": "Иллюстрация исхода", "contents": [{
                "type": "image", "base64": base64.b64encode(Path(image).read_bytes()).decode("ascii"),
                "mime_type": mime_types[path.suffix.lower()],
            }]}]
        self.status = "Газета с иллюстрацией" if image else "Газета без иллюстрации: подходящая картинка не загружена"
        return Message(text=result["herald_text"], files=files, content_blocks=blocks)

    def build_herald_prompt(self) -> Message:
        return Message(text=self._forecast()["herald_prompt"])
