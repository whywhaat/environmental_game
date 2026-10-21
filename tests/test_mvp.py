import json
import io
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from negotiation_mvp.cli import main
from negotiation_mvp.core import GameConfig, GameError, State, merge_round, parse_sets, render_table, score_state
from negotiation_mvp.engine import ConversionError, _get_vote, _round_notice, convert_speech, run_game
from negotiation_mvp.llm import OpenAICompatibleChat
from negotiation_mvp.report import render_report


CONFIG = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "example_config.json").read_text(encoding="utf-8")))


class ScriptedChat:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete(self, model, messages):
        self.calls.append((model, messages))
        return next(self.replies)


def sets(speaker, source, seq=1):
    return parse_sets(source, speaker, CONFIG, seq)


class CoreTests(unittest.TestCase):
    def test_cost_and_validation(self):
        self.assertEqual(CONFIG.total_cost, 270_000_000)
        with self.assertRaises(GameError):
            sets("industrial", 'SET components.unknown.responsible = "industrial"')
        with self.assertRaises(GameError):
            sets("industrial", "SET participants.administration.contribution = -1")
        with self.assertRaises(GameError):
            sets("industrial", "SET participants.administration.vote = true")
        with self.assertRaises(GameError):
            sets("industrial", "SET components.timeline.months = 12")

    def test_old_value_conflicts_and_conflict_has_no_memory(self):
        path = "components.water_supply.responsible"
        first = merge_round(State(), sets("industrial", 'SET components.water_supply.responsible = "industrial"'), CONFIG)
        repeated = merge_round(first, sets("industrial", 'SET components.water_supply.responsible = "industrial"', 2), CONFIG)
        self.assertFalse(repeated.conflict(path))
        self.assertEqual(len(repeated.records(path)), 1)
        changed = merge_round(repeated, sets("administration", 'SET components.water_supply.responsible = "administration"', 3), CONFIG)
        self.assertTrue(changed.conflict(path))
        self.assertEqual([r["value"] for r in changed.records(path)], ["industrial", "administration"])
        self.assertIn("Администрация", render_table(changed, CONFIG))
        resolved = merge_round(changed, sets("industrial", 'SET components.water_supply.responsible = "industrial"', 4), CONFIG)
        self.assertEqual(resolved.value(path), "industrial")
        self.assertEqual(len(resolved.records(path)), 1)

    def test_same_round_conflict_and_missing_finance_scores_zero(self):
        state = merge_round(State(), sets("administration", "SET participants.administration.contribution = 50000000") + sets("industrial", "SET participants.administration.contribution = 10000000", 2), CONFIG)
        self.assertTrue(state.conflict("participants.administration.contribution"))
        self.assertEqual(score_state(state, CONFIG)["administration"], 0)

    def test_table_shows_score_and_credit(self):
        state = merge_round(State(), sets("administration", "SET participants.administration.contribution = 20000000"), CONFIG)
        table = render_table(state, CONFIG)
        self.assertIn("| Оценка: Администрация | -2 | — |", table)
        self.assertIn("| Кредит: Администрация | 10 млн ₽ | — |", table)
        self.assertIn("| Кредит: Промкомплекс | — | — |", table)

    def test_three_round_example(self):
        replies = [
            "Здравствуйте, представители промкомплекса, присаживайтесь.",
            "NO_CHANGES",
            "Предлагаем 50 млн от нас и 220 млн от администрации.",
            'SET participants.industrial.contribution = 50000000\nSET participants.administration.contribution = 220000000',
            "При бюджете 10 млн предлагаю наш взнос 50 млн, ваш 220 млн; воду ведём мы, мониторинг и реконструкцию — вы.",
            'SET participants.administration.contribution = 50000000\nSET participants.industrial.contribution = 220000000\nSET components.water_supply.responsible = "administration"\nSET components.independent_monitoring.responsible = "industrial"\nSET components.reconstruction.responsible = "industrial"',
            "Предлагаю ваш взнос 10 млн, наш 260 млн; воду организуете вы, мониторинг и реконструкцию — мы.",
            'SET participants.administration.contribution = 10000000\nSET participants.industrial.contribution = 260000000\nSET components.water_supply.responsible = "administration"\nSET components.independent_monitoring.responsible = "industrial"\nSET components.reconstruction.responsible = "industrial"',
            "Согласны: администрация 10 млн, промкомплекс 260 млн.",
            "SET participants.administration.contribution = 10000000\nSET participants.industrial.contribution = 260000000",
            "Подтверждаем условия.",
            "NO_CHANGES",
            "VOTE ЗА — поддерживаем договор.",
            "VOTE ЗА — поддерживаем договор.",
        ]
        chat = ScriptedChat(replies)
        result = run_game(CONFIG, chat, "player", "converter")
        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual([r["scores"] for r in result["rounds"]], [
            {"administration": -22, "industrial": -1},
            {"administration": 1, "industrial": -1},
            {"administration": 0, "industrial": -6.2},
        ])
        self.assertEqual(len(result["rounds"][1]["conflicts"]), 2)
        self.assertEqual(result["rounds"][2]["conflicts"], [])
        self.assertEqual(result["rounds"][2]["funding_total"], 270_000_000)
        self.assertEqual(result["final_state"]["fields"]["components.water_supply.responsible"][0]["value"], "administration")
        self.assertTrue(result["accepted"])
        self.assertEqual(chat.calls[0][1][0]["content"], CONFIG.players[0].personal_context)
        first_player_messages = chat.calls[0][1]
        self.assertIn({"role": "user", "content": result["session"][1]["text"]}, first_player_messages)
        self.assertIn("Отрицательная оценка допустима", result["session"][1]["text"])
        self.assertIn({"role": "user", "content": result["session"][2]["text"]}, first_player_messages)
        self.assertTrue(any(f"Раунд 1 из {CONFIG.max_rounds}" in message["content"] for message in first_player_messages))
        admin_expectations = next(message["content"] for message in first_player_messages if message["content"].startswith("Личные ожидания для оценки:"))
        self.assertIn("желаемый собственный взнос — 0%", admin_expectations)
        self.assertIn("Вода и заморозка тарифов — Промкомплекс", admin_expectations)
        self.assertNotIn("срок", admin_expectations)
        self.assertTrue(any("Финансовый отдел подтвердил" in message["content"] for message in first_player_messages))
        self.assertFalse(any("Руководство выделило" in message["content"] for message in first_player_messages))
        second_player_messages = chat.calls[2][1]
        industrial_expectations = next(message["content"] for message in second_player_messages if message["content"].startswith("Личные ожидания для оценки:"))
        self.assertIn("Вода и заморозка тарифов — Администрация", industrial_expectations)
        self.assertNotIn("срок", industrial_expectations)
        self.assertNotIn(admin_expectations, [message["content"] for message in second_player_messages])
        self.assertFalse(any("Личные ожидания для оценки:" in event["text"] for event in result["session"]))
        self.assertTrue(any("Руководство выделило" in message["content"] for message in second_player_messages))
        self.assertFalse(any("Финансовый отдел подтвердил" in message["content"] for message in second_player_messages))
        self.assertFalse(any("Личная история:" in event["text"] for event in result["session"]))
        second_round_messages = chat.calls[4][1]
        self.assertIn({"role": "user", "content": result["rounds"][0]["table"]}, second_round_messages)
        self.assertFalse(any("Текущая таблица STATE" in message["content"] for message in second_round_messages))
        self.assertEqual([event["kind"] for event in result["session"]].count("table"), 4)
        self.assertEqual([event["kind"] for event in result["session"]].count("clock"), 3)
        report = render_report(result, CONFIG)
        self.assertIn("## Раунд 3", report)
        self.assertIn("**Извлечённые SET:**", report)
        self.assertIn("**Решение:** договор принят.", report)
        self.assertIn("Промкомплекс -6,2", report)

    def test_round_notices_escalate_without_ending_early(self):
        self.assertIn("осталось 15 раундов", _round_notice(1, 15))
        self.assertIn("Время на согласование заканчивается", _round_notice(13, 15))
        self.assertIn("последний раунд переговоров", _round_notice(15, 15))

    def test_bad_converter_retries_then_fails(self):
        chat = ScriptedChat(['SET participants.industrial.contribution = "50000000"'] * 2)
        with self.assertRaises(ConversionError):
            convert_speech(chat, "converter", "Предлагаю 50 млн.", "industrial", CONFIG, 1)
        self.assertEqual(len(chat.calls), 2)

    def test_unsupported_contribution_is_discarded_after_retry(self):
        output = 'SET participants.administration.contribution = 10000000\nSET components.water_supply.responsible = "industrial"\nSET participants.industrial.contribution = 12000000'
        chat = ScriptedChat([output, output])
        applied, actions = convert_speech(chat, "converter", "Администрация вносит 10 млн рублей, за воду отвечает Промкомплекс.", "administration", CONFIG, 1)
        self.assertEqual(len(chat.calls), 2)
        self.assertNotIn("industrial.contribution", applied)
        self.assertEqual([action.path for action in actions], ["participants.administration.contribution", "components.water_supply.responsible"])

    def test_computed_score_set_is_excluded_by_schema(self):
        output = 'SET components.water_supply.responsible = "administration"\nSET participants.industrial.contribution = 0\nSET participants.industrial.score = 3'
        chat = ScriptedChat([output, output])
        applied, actions = convert_speech(chat, "converter", "Воду ведёт администрация, наш взнос 0 рублей.", "industrial", CONFIG, 1)
        self.assertEqual(len(chat.calls), 2)
        self.assertNotIn(".score", applied)
        self.assertEqual([action.path for action in actions], ["components.water_supply.responsible", "participants.industrial.contribution"])
        self.assertNotIn("participants.industrial.score", merge_round(State(), actions, CONFIG).fields)

    def test_component_contributions_are_excluded_by_schema(self):
        output = (
            'SET components.water_supply.responsible = "industrial"\n'
            'SET components.water_supply.contribution = 50000000\n'
            'SET components.independent_monitoring.responsible = "industrial"\n'
            'SET components.independent_monitoring.contribution = 20000000\n'
            'SET components.reconstruction.responsible = "administration"\n'
            'SET components.reconstruction.contribution = 200000000'
        )
        chat = ScriptedChat([output, output])
        applied, actions = convert_speech(chat, "converter", "За воду и мониторинг отвечаем мы, за реконструкцию администрация.", "industrial", CONFIG, 1)
        self.assertEqual(len(chat.calls), 2)
        self.assertNotIn(".contribution", applied)
        self.assertEqual([action.path for action in actions], [
            "components.water_supply.responsible",
            "components.independent_monitoring.responsible",
            "components.reconstruction.responsible",
        ])
        invalid = ScriptedChat(["SET components.water_supply.other = 1"] * 2)
        applied, actions = convert_speech(invalid, "converter", "Предлагаю обсудить воду.", "industrial", CONFIG, 1)
        self.assertEqual((applied, actions), ("NO_CHANGES", []))
        corrected = ScriptedChat(["SET components.water_supply.contribution = 50000000", "SET participants.industrial.contribution = 50000000"])
        applied, actions = convert_speech(corrected, "converter", "Промкомплекс внесёт 50 млн рублей.", "industrial", CONFIG, 1)
        self.assertEqual(applied, "SET participants.industrial.contribution = 50000000")
        self.assertEqual(len(actions), 1)

    def test_conditional_vote_counts_as_against(self):
        chat = ScriptedChat(["Если вы примете мои условия, я подпишу договор."])
        self.assertFalse(_get_vote(chat, "player", CONFIG.players[0], CONFIG, [])[1])

    def test_openai_compatible_http_client(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((self.path, self.headers["Authorization"], json.loads(body)))
                reply = json.dumps({"choices": [{"message": {"content": "NO_CHANGES"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = OpenAICompatibleChat(f"http://127.0.0.1:{server.server_port}/v1", "test-key")
            self.assertEqual(client.complete("test-model", [{"role": "user", "content": "hello"}]), "NO_CHANGES")
            self.assertEqual(received[0][0], "/v1/chat/completions")
            self.assertEqual(received[0][1], "Bearer test-key")
            self.assertEqual(received[0][2]["model"], "test-model")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_chat_client_retries_one_timeout(self):
        reply = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Готово"}}]}).encode("utf-8"))
        client = OpenAICompatibleChat("http://127.0.0.1:1/v1", timeout=1)
        with patch("urllib.request.urlopen", side_effect=[TimeoutError("timed out"), reply]) as request:
            self.assertEqual(client.complete("player", [{"role": "user", "content": "Ход"}]), "Готово")
            self.assertEqual(request.call_count, 2)

    def test_cli_runs_against_compatible_http_endpoint(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = "\n".join(message["content"] for message in request["messages"])
                if request["model"] == "player":
                    answer = "VOTE ЗА" if "Голосование по договору открыто." in prompt else "Пока предложений нет."
                else:
                    answer = "NO_CHANGES"
                reply = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output = Path(temp_dir) / "result.json"
                env = {"OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "OPENAI_API_KEY": "test-key", "PLAYER_MODEL": "player", "CONVERTER_MODEL": "converter"}
                with patch.dict("os.environ", env):
                    self.assertEqual(main(["--config", str(Path(__file__).resolve().parents[1] / "example_config.json"), "--output", str(output), "--games-dir", temp_dir]), 0)
                output = next(Path(temp_dir).glob("*/result.json"))
                result = json.loads(output.read_text(encoding="utf-8"))
                report = output.with_suffix(".md")
                prompt_report = output.with_name("result_with_prompts.md")
                self.assertTrue(report.exists())
                self.assertIn("**Решение:** договор не принят.", report.read_text(encoding="utf-8"))
                self.assertTrue(prompt_report.exists())
                extended = prompt_report.read_text(encoding="utf-8")
                self.assertTrue(extended.startswith(report.read_text(encoding="utf-8").rstrip()))
                self.assertIn("## Сообщения, переданные LLM", extended)
                self.assertIn(CONFIG.players[0].personal_context, extended)
                self.assertIn("Ты конвертер реплик", extended)
                self.assertEqual(len(result["rounds"]), CONFIG.max_rounds)
                self.assertEqual(result["votes"], {"administration": True, "industrial": True})
                self.assertFalse(result["accepted"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_interrupted_run_saves_prompt_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.json"
            env = {"OPENAI_BASE_URL": "http://127.0.0.1:1/v1", "PLAYER_MODEL": "player", "CONVERTER_MODEL": "converter"}
            with patch.dict("os.environ", env), patch("negotiation_mvp.cli.run_game", side_effect=RuntimeError("converter failed")):
                with self.assertRaisesRegex(RuntimeError, "converter failed"):
                    main(["--config", str(Path(__file__).resolve().parents[1] / "example_config.json"), "--output", str(output), "--games-dir", temp_dir])
            output = next(Path(temp_dir).glob("*/result.md")).with_suffix(".json")
            self.assertFalse(output.exists())
            self.assertIn("запуск технически прерван", output.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertIn("## Сообщения, переданные LLM", output.with_name("result_with_prompts.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
