"""Проверка отмены на настоящих LFX и ChatOpenAI через локальный HTTP-сбой.

Запускать Python из окружения Langflow. Модели не подменяются; GigaChat не
вызывается. Локальный сервер намеренно отвечает 503 либо не отвечает, чтобы
проверить транспорт, число попыток, отмену и сохранение отчёта без расхода квоты.
"""

import asyncio
import contextlib
import inspect
import json
import os
import tempfile
from pathlib import Path

from langchain_openai import ChatOpenAI
from lfx.custom.eval import eval_custom_component_code
from lfx.schema.data import Data


ROOT = Path(__file__).resolve().parents[1]


async def verify():
    component = eval_custom_component_code(
        (ROOT / "langflow/custom_components/my_components/dialog.py").read_text())
    assert inspect.iscoroutinefunction(component.run_chat)
    mode = "fail"
    calls = []
    started = asyncio.Event()
    handlers = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        handlers.add(task)
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next((int(line.split(b":", 1)[1]) for line in headers.split(b"\r\n")
                           if line.lower().startswith(b"content-length:")), 0)
            await reader.readexactly(length)
            calls.append(1)
            started.set()
            if mode == "fail":
                body = b'{"error":{"message":"controlled transport failure","type":"server_error"}}'
                writer.write(b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\n"
                             + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
                await writer.drain()
            else:
                await reader.read()  # Ждём закрытия соединения отменённым клиентом.
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    def game():
        model = ChatOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="diagnostic-only",
                           model="transport-check", max_retries=4)
        instance = component()
        instance.players = [Data(data={"id": "city_administration", "name": "Администрация",
                            "budget": 100_000_000, "prompt": "Проверка транспорта", "llm": model})]
        instance.request_timeout_seconds = 1
        instance.request_attempts = 2
        instance.game_timeout_seconds = 10
        return instance

    previous = os.environ.get("NEGOTIATION_GAMES_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix="ecolog-lf-lifecycle-") as directory:
            os.environ["NEGOTIATION_GAMES_DIR"] = directory
            instance = game()
            result = await asyncio.wait_for(instance.run_chat(), 8)
            assert result.data["status"] == "failed"
            assert len(calls) == 2, f"Неожиданные SDK-повторы: {len(calls)}"
            assert Path(result.data["files"]["result.md"]).exists()
            report_node = next(node for node in json.loads(
                (ROOT / "langflow/Negotiation_MVP_5_players.json").read_text())["data"]["nodes"]
                if node["data"]["type"] == "ReportController")
            report_class = eval_custom_component_code(report_node["data"]["node"]["template"]["code"]["value"])
            report = report_class()
            report.game_result = result
            assert "Игра прервана" in report.generate_report().text
            print("REAL_STACK: HTTP 503 -> ровно 2 запроса, частичный отчёт возвращён и сохранён", flush=True)

            mode = "hang"
            calls.clear()
            started.clear()
            instance = game()
            task = asyncio.create_task(instance.run_chat())
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Отмена была подавлена")
            await asyncio.sleep(2.6)  # Дольше возможной первой паузы повторного запроса.
            assert len(calls) == 1
            saved = json.loads((instance._run_store.directory / "result.json").read_text())
            assert saved["status"] == "cancelled"
            print("REAL_STACK: отмена -> соединение отменено, новых запросов нет, отчёт cancelled сохранён", flush=True)

            calls.clear()
            instance = game()
            instance.request_attempts = 1
            result = await asyncio.wait_for(instance.run_chat(), 5)
            assert result.data["status"] == "failed"
            assert len(calls) == 1
            assert "не ответила" in result.data["error"]
            print("REAL_STACK: молчащий сервер -> лимит запроса сработал, отчёт failed сохранён", flush=True)

            consequences_code = (ROOT / "langflow/custom_components/my_components/ConsequencesModel.py").read_text()
            consequences = eval_custom_component_code(consequences_code)()
            consequences.game_result = result
            assert "не завершена" in consequences.build_report().text
            assert consequences.build_forecast().data["status"] == "skipped"
            newspaper = consequences.build_newspaper()
            assert "не завершена" in newspaper.text
            assert not newspaper.files and not newspaper.content_blocks
            print("REAL_STACK: ReportController отображает ошибку; незавершённая игра не даёт ложный прогноз", flush=True)
    finally:
        if previous is None:
            os.environ.pop("NEGOTIATION_GAMES_DIR", None)
        else:
            os.environ["NEGOTIATION_GAMES_DIR"] = previous
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(verify())
