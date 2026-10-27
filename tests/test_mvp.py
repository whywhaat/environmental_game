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
from negotiation_mvp.engine import COMMON_PLAYER_SYSTEM_PROMPT, ConversionError, _normalize_converter_sets, _round_notice, convert_speech, run_game
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

    def test_conflicts_persist_across_rounds_until_consensus(self):
        path = "components.water_supply.responsible"
        first = merge_round(State(), sets("industrial", 'SET components.water_supply.responsible = "industrial"'), CONFIG)
        repeated = merge_round(first, sets("industrial", 'SET components.water_supply.responsible = "industrial"', 2), CONFIG)
        self.assertFalse(repeated.conflict(path))
        self.assertEqual(len(repeated.records(path)), 1)
        changed = merge_round(repeated, sets("administration", 'SET components.water_supply.responsible = "administration"', 3), CONFIG)
        self.assertTrue(changed.conflict(path))
        self.assertEqual([r["value"] for r in changed.records(path)], ["industrial", "administration"])
        self.assertIn("Администрация", render_table(changed, CONFIG))
        # Conflict does not magically disappear when industrial merely re-asserts its proposal
        still_conflicted = merge_round(changed, sets("industrial", 'SET components.water_supply.responsible = "industrial"', 4), CONFIG)
        self.assertTrue(still_conflicted.conflict(path))
        self.assertEqual(len(still_conflicted.records(path)), 2)
        # Conflict resolves only when administration accepts or agrees
        resolved = merge_round(still_conflicted, sets("administration", 'SET components.water_supply.responsible = "industrial"', 5), CONFIG)
        self.assertFalse(resolved.conflict(path))
        self.assertEqual(resolved.value(path), "industrial")
        self.assertEqual(len(resolved.records(path)), 2)

    def test_same_round_conflict_and_missing_finance_has_shared_penalty(self):
        state = merge_round(State(), sets("administration", "SET participants.administration.contribution = 50000000") + sets("industrial", "SET participants.administration.contribution = 10000000", 2), CONFIG)
        self.assertTrue(state.conflict("participants.administration.contribution"))
        self.assertEqual(score_state(state, CONFIG)["administration"], 5)

    def test_shared_funding_score_is_capped_at_full_cost(self):
        partial = merge_round(State(), sets("administration", "SET participants.administration.contribution = 20000000"), CONFIG)
        full = merge_round(State(), sets("administration", "SET participants.administration.contribution = 270000000"), CONFIG)
        excess = merge_round(State(), sets("administration", "SET participants.administration.contribution = 300000000"), CONFIG)
        self.assertAlmostEqual(score_state(partial, CONFIG)["industrial"], 50 - 10 * 250 / 60)
        self.assertEqual(score_state(full, CONFIG)["industrial"], 50)
        self.assertEqual(score_state(excess, CONFIG)["industrial"], 50)
        self.assertEqual(score_state(full, CONFIG, signed=True)["industrial"], 100)
        self.assertEqual(score_state(excess, CONFIG)["administration"], 0)

    def test_self_responsibility_allowed_and_rewarded(self):
        raw = json.loads((Path(__file__).resolve().parents[1] / "example_config.json").read_text(encoding="utf-8"))
        raw["players"][0]["desired_responsible"]["water_supply"] = "administration"
        config = GameConfig.from_dict(raw)
        self.assertEqual(config.players[0].desired_responsible["water_supply"], "administration")

        state = merge_round(State(), parse_sets('SET components.water_supply.responsible = "administration"', "administration", config), config)
        empty_state = State()
        self.assertEqual(score_state(state, config, signed=True)["administration"], score_state(empty_state, config, signed=True)["administration"] + 10)

        state_ind = merge_round(State(), parse_sets('SET components.water_supply.responsible = "industrial"', "industrial", config), config)
        self.assertEqual(score_state(state_ind, config, signed=True)["industrial"], score_state(empty_state, config, signed=True)["industrial"] - 10)

    def test_yes_votes_do_not_lift_penalty_without_complete_contract(self):
        chat = ScriptedChat((["Без изменений", "NO_CHANGES"] * (len(CONFIG.players) * CONFIG.max_rounds)) + [
            "Я за договор.", "SET participants.administration.vote = true",
            "Поддерживаю договор.", "SET participants.industrial.vote = true",
        ])
        result = run_game(CONFIG, chat, "player", "converter")
        self.assertEqual(result["votes"], {player.id: True for player in CONFIG.players})
        self.assertFalse(result["accepted"])
        self.assertEqual(result["final_scores"], {player.id: 5 for player in CONFIG.players})
        self.assertIn("| Оценка: Администрация | 5 | — |", result["final_table"])

    def test_missing_final_vote_is_abstention(self):
        chat = ScriptedChat((["Без изменений", "NO_CHANGES"] * (len(CONFIG.players) * CONFIG.max_rounds)) + [
            "Если условия изменятся, подпишу.", "NO_CHANGES",
            "Я против договора.", "SET participants.industrial.vote = false",
        ])
        result = run_game(CONFIG, chat, "player", "converter")
        self.assertEqual(result["votes"], {"administration": None, "industrial": False})
        self.assertNotIn("participants.administration.vote", result["final_state"]["fields"])
        self.assertIn("### Администрация — воздержался", render_report(result, CONFIG))

    def test_table_shows_score_and_credit(self):
        state = merge_round(State(), sets("administration", "SET participants.administration.contribution = 20000000"), CONFIG)
        table = render_table(state, CONFIG)
        self.assertIn("| Оценка: Администрация | 0 | — |", table)
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
            "Поддерживаю договор.",
            "SET participants.administration.vote = true",
            "Голосую за.",
            "SET participants.industrial.vote = true",
        ]
        chat = ScriptedChat(replies)
        result = run_game(CONFIG, chat, "player", "converter")
        self.assertEqual(len(result["rounds"]), 3)
        self.assertEqual([r["scores"] for r in result["rounds"]], [
            {"administration": 0, "industrial": 40},
            {"administration": 15, "industrial": 0},
            {"administration": 50, "industrial": 0},
        ])
        self.assertEqual(len(result["rounds"][1]["conflicts"]), 2)
        self.assertEqual(result["rounds"][2]["conflicts"], [])
        self.assertEqual(result["rounds"][2]["funding_total"], 270_000_000)
        self.assertEqual(result["final_state"]["fields"]["components.water_supply.responsible"][0]["value"], "administration")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["final_scores"], {"administration": 100, "industrial": 38})
        self.assertIn("| Оценка: Администрация | 100 | — |", result["final_table"])
        self.assertEqual(chat.calls[0][1][0]["content"], COMMON_PLAYER_SYSTEM_PROMPT + "\n\n" + CONFIG.players[0].personal_context)
        self.assertIn("Если твой бюджет равен нулю", chat.calls[0][1][0]["content"])
        self.assertIn("общая часть за финансирование", chat.calls[0][1][0]["content"])
        self.assertIn("штраф за неподписанный договор", chat.calls[0][1][0]["content"])
        self.assertIn("Говори разговорным, неформальным языком", chat.calls[0][1][0]["content"])
        first_player_messages = chat.calls[0][1]
        self.assertIn({"role": "user", "content": result["session"][1]["text"]}, first_player_messages)
        self.assertIn("Низкая оценка отражает", result["session"][1]["text"])
        self.assertIn({"role": "user", "content": result["session"][2]["text"]}, first_player_messages)
        self.assertEqual(first_player_messages[-1], {"role": "user", "content": "Раунд 1. Слово предоставлено: Администрация."})
        self.assertEqual(sum(message["role"] == "system" for message in first_player_messages), 1)
        self.assertNotIn({"role": "user", "content": first_player_messages[0]["content"]}, first_player_messages)
        self.assertEqual(sum(message["content"].startswith("Личные ожидания для оценки:") for message in first_player_messages), 1)
        self.assertEqual(sum(message["content"].startswith("Оценка каждого участника:") for message in first_player_messages), 1)
        self.assertEqual(sum(message["content"].startswith("Слушания открыты.") for message in first_player_messages), 1)
        self.assertEqual(sum(message["content"].startswith("Раунд 1 из") for message in first_player_messages), 1)
        self.assertTrue(any(f"Раунд 1 из {CONFIG.max_rounds}" in message["content"] for message in first_player_messages))
        admin_expectations = next(message["content"] for message in first_player_messages if message["content"].startswith("Личные ожидания для оценки:"))
        self.assertIn("желаемый собственный взнос — 0%", admin_expectations)
        self.assertIn("Вода и заморозка тарифов — Промкомплекс", admin_expectations)
        self.assertNotIn("срок", admin_expectations)
        self.assertTrue(any("Финансовый отдел подтвердил" in message["content"] for message in first_player_messages))
        self.assertFalse(any("Руководство выделило" in message["content"] for message in first_player_messages))
        second_player_messages = chat.calls[2][1]
        self.assertEqual(second_player_messages[0]["content"], COMMON_PLAYER_SYSTEM_PROMPT + "\n\n" + CONFIG.players[1].personal_context)
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
        self.assertIn("Промкомплекс 38", report)

    def test_round_notices_escalate_without_ending_early(self):
        self.assertIn("осталось 15 раундов", _round_notice(1, 15))
        self.assertIn("Время на согласование заканчивается", _round_notice(13, 15))
        self.assertIn("последний раунд переговоров", _round_notice(15, 15))

    def test_bad_converter_retries_then_fails(self):
        chat = ScriptedChat(['SET participants.industrial.contribution = "пятьдесят миллионов"'] * 2)
        with self.assertRaises(ConversionError):
            convert_speech(chat, "converter", "Предлагаю 50 млн.", "industrial", CONFIG, 1)
        self.assertEqual(len(chat.calls), 2)
        for _, messages in chat.calls:
            self.assertEqual([message["role"] for message in messages], ["system", "user"])
            self.assertEqual(messages[0]["content"].count("Ты конвертер реплик"), 2)
            self.assertEqual(messages[1]["content"].count("Автор реплики:"), 2)
        self.assertIn('Предыдущий ответ:\nSET participants.industrial.contribution = "пятьдесят миллионов"', chat.calls[1][1][1]["content"])

    def test_converter_omits_absent_responsible_without_losing_other_sets(self):
        chat = ScriptedChat([
            'SET components.water_supply.responsible = "Администрация"\n'
            'SET components.reconstruction.responsible = "NO_CHANGES"'
        ])
        applied, actions = convert_speech(chat, "converter", "За воду отвечает Администрация, реконструкция — мимо.", "industrial", CONFIG, 1)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(applied, 'SET components.water_supply.responsible = "administration"')
        self.assertEqual([action.value for action in actions], ["administration"])

        empty = ScriptedChat(['SET components.reconstruction.responsible = "мимо"'])
        self.assertEqual(convert_speech(empty, "converter", "Реконструкция — мимо.", "industrial", CONFIG, 1), ("NO_CHANGES", []))

    def test_converter_normalizes_responsible_player_names_and_bare_ids(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        chat = ScriptedChat([
            'SET components.water_supply.responsible = industrial_holding\n'
            'SET components.reconstruction.responsible = "промкомплекс"'
        ])
        applied, actions = convert_speech(chat, "converter", "За воду и реконструкцию отвечает Промкомплекс.", "city_administration", config, 1)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual([message["role"] for message in chat.calls[0][1]], ["system", "user"])
        self.assertEqual(chat.calls[0][1][1]["content"].count("Автор реплики:"), 2)
        self.assertEqual([action.value for action in actions], ["industrial_holding", "industrial_holding"])
        self.assertEqual(applied, 'SET components.water_supply.responsible = "industrial_holding"\nSET components.reconstruction.responsible = "industrial_holding"')

    def test_converter_uses_configured_id_example_and_unambiguous_short_id(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        raw = (
            'SET participants.city_administration.contribution = 0\n'
            'SET components.water_supply.responsible = "industrial"\n'
            'SET components.independent_monitoring.responsible = "regional_government"\n'
            'SET components.reconstruction.responsible = "industrial"'
        )
        chat = ScriptedChat([raw])
        applied, actions = convert_speech(
            chat, "converter",
            "Администрация вносит 0 рублей; за воду и реконструкцию отвечает Промкомплекс, за мониторинг — Регион.",
            "city_administration", config, 1,
        )
        self.assertEqual(len(chat.calls), 1)
        self.assertIn('например SET components.water_supply.responsible = "city_administration"', chat.calls[0][1][1]["content"])
        self.assertEqual([action.value for action in actions], [0, "industrial_holding", "regional_government", "industrial_holding"])
        self.assertNotIn('= "industrial"', applied)
        self.assertEqual(_normalize_converter_sets('SET components.water_supply.responsible = "city"', config), 'SET components.water_supply.responsible = "city"')

    def test_converter_normalizes_all_text_values_without_changing_numbers_or_booleans(self):
        raw = "SET participants.industrial.note = решение принято\nSET participants.industrial.contribution = 50000000\nSET participants.industrial.vote = true"
        self.assertEqual(
            _normalize_converter_sets(raw, CONFIG),
            'SET participants.industrial.note = "решение принято"\nSET participants.industrial.contribution = 50000000\nSET participants.industrial.vote = true',
        )

    def test_converter_normalizes_each_supported_value_type(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        raw = (
            'SET participants.regional_government.contribution = "10000000"\n'
            'SET components.water_supply.responsible = "industrial"\n'
            'SET participants.city_administration.vote = "за"'
        )
        self.assertEqual(
            _normalize_converter_sets(raw, config),
            'SET participants.regional_government.contribution = 10000000\n'
            'SET components.water_supply.responsible = "industrial_holding"\n'
            'SET participants.city_administration.vote = true',
        )
        self.assertEqual(
            _normalize_converter_sets('SET participants.regional_government.contribution = "10 млн ₽"', config),
            'SET participants.regional_government.contribution = 10000000',
        )
        self.assertEqual(
            _normalize_converter_sets('SET participants.regional_government.contribution = "10 000 000 ₽"', config),
            'SET participants.regional_government.contribution = 10000000',
        )
        self.assertEqual(
            _normalize_converter_sets('SET participants.regional_government.contribution = "2,5 млн ₽"', config),
            'SET participants.regional_government.contribution = 2500000',
        )
        self.assertEqual(
            _normalize_converter_sets('SET participants.regional_government.contribution = "неизвестно"', config),
            'SET participants.regional_government.contribution = "неизвестно"',
        )

    def test_converter_ignores_no_changes_line_beside_valid_set(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        raw = "SET participants.regional_government.contribution = 20000000\nNO_CHANGES"
        chat = ScriptedChat([raw])
        applied, actions = convert_speech(chat, "converter", "Регион вносит 20 млн рублей.", "industrial_holding", config, 1)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(applied, "SET participants.regional_government.contribution = 20000000")
        self.assertEqual([action.value for action in actions], [20_000_000])
        self.assertEqual(_normalize_converter_sets("NO_CHANGES\nNO_CHANGES", config), "NO_CHANGES")

    def test_quoted_contribution_from_report_is_accepted_on_first_try(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        raw = (
            'SET participants.regional_government.contribution = "10000000"\n'
            'SET components.water_supply.responsible = "city_administration"\n'
            'SET components.independent_monitoring.responsible = "regional_government"\n'
            'SET components.reconstruction.responsible = "industrial_holding"'
        )
        chat = ScriptedChat([raw])
        applied, actions = convert_speech(
            chat, "converter",
            "Регион вносит 10 млн рублей. Воду ведёт Администрация, мониторинг — Регион, реконструкцию — Промкомплекс.",
            "industrial_holding", config, 1,
        )
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(actions[0].value, 10_000_000)
        self.assertIn("contribution = 10000000", applied)

    def test_duplicate_component_costs_do_not_become_a_player_contribution(self):
        config = GameConfig.from_dict(json.loads((Path(__file__).resolve().parents[1] / "langflow" / "demo_config.json").read_text(encoding="utf-8")))
        raw = (
            'SET components.water_supply.responsible = "industrial_holding"\n'
            'SET participants.industrial_holding.contribution = 50000000\n'
            'SET components.independent_monitoring.responsible = "industrial_holding"\n'
            'SET participants.industrial_holding.contribution = 20000000\n'
            'SET components.reconstruction.responsible = "city_administration"'
        )
        speech = (
            "Воду и заморозку тарифов отдаю Промкомплексу (это 50 млн), пусть они отвечают за качество. "
            "Независимый мониторинг тоже пусть берёт Промкомплекс (20 млн). "
            "Реконструкцию оставляю за собой. Наш взнос — 0%, кто сколько готов вложиться — обсудим."
        )
        chat = ScriptedChat([raw, raw])
        applied, actions = convert_speech(chat, "converter", speech, "city_administration", config, 1)
        self.assertEqual(len(chat.calls), 2)
        self.assertNotIn("contribution", applied)
        self.assertEqual([action.path for action in actions], [
            "components.water_supply.responsible",
            "components.independent_monitoring.responsible",
            "components.reconstruction.responsible",
        ])

    def test_identical_duplicate_contribution_is_kept_once(self):
        raw = 'SET participants.industrial.contribution = 50000000\nSET participants.industrial.contribution = 50000000'
        chat = ScriptedChat([raw, raw])
        applied, actions = convert_speech(chat, "converter", "Промкомплекс вносит 50 млн рублей.", "industrial", CONFIG, 1)
        self.assertEqual((applied, [action.value for action in actions]), ("SET participants.industrial.contribution = 50000000", [50_000_000]))

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

    def test_converter_extracts_natural_votes_and_preserves_abstention(self):
        for speech, output, expected in (
            ('Голосую "ЗА"!', "SET participants.administration.vote = true", True),
            ("Я против договора.", "SET participants.administration.vote = false", False),
            ("Если вы примете мои условия, я подпишу договор.", "NO_CHANGES", None),
        ):
            chat = ScriptedChat([output])
            applied, actions = convert_speech(chat, "converter", speech, "administration", CONFIG, 1, voting=True)
            self.assertEqual(applied, output)
            self.assertEqual(actions[0].value if actions else None, expected)
            self.assertIn("Сейчас финальное голосование", chat.calls[0][1][1]["content"])

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
                    answer = 'Голосую "ЗА"!' if "Голосование по договору открыто." in prompt else "Пока предложений нет."
                elif "Сейчас финальное голосование" in prompt:
                    speaker = prompt.split("Автор реплики: ", 1)[1].splitlines()[0]
                    answer = f"SET participants.{speaker}.vote = true"
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


    def test_parameter_levers_parsing_validation_and_merge(self):
        sets_text = (
            'SET parameters.guilt = "shared"\n'
            'SET parameters.plant_mode = "70%"\n'
            'SET parameters.tariffs_freeze = true\n'
            'SET parameters.timeline_months = 24'
        )
        actions = parse_sets(sets_text, "administration", CONFIG)
        self.assertEqual(len(actions), 4)
        self.assertEqual(actions[0].path, "parameters.guilt")
        self.assertEqual(actions[0].value, "shared")
        self.assertEqual(actions[1].value, "70%")
        self.assertEqual(actions[2].value, True)
        self.assertEqual(actions[3].value, 24)

        # Invalid param or value raises GameError
        with self.assertRaises(GameError):
            parse_sets('SET parameters.unknown_lever = "val"', "administration", CONFIG)
        with self.assertRaises(GameError):
            parse_sets('SET parameters.plant_mode = "50%"', "administration", CONFIG)

        # Merge round records parameters and detects conflict
        state = merge_round(State(), actions, CONFIG)
        self.assertEqual(state.value("parameters.guilt"), "shared")
        self.assertEqual(state.value("parameters.plant_mode"), "70%")
        self.assertFalse(state.conflict("parameters.plant_mode"))

        # Conflicting proposal from industrial
        conflict_actions = parse_sets('SET parameters.plant_mode = "100%"', "industrial", CONFIG)
        state2 = merge_round(state, conflict_actions, CONFIG)
        self.assertTrue(state2.conflict("parameters.plant_mode"))
        self.assertIn("parameters.plant_mode", state2.conflicts())
        self.assertIsNone(state2.value("parameters.plant_mode"))

    def test_parameter_normalization_in_converter(self):
        raw = (
            'SET parameters.guilt = "паритет"\n'
            'SET parameters.tariffs_freeze = "да"\n'
            'SET parameters.tax_relief = "нет"\n'
            'SET parameters.plant_mode = "70"\n'
            'SET parameters.timeline_months = "24 месяца"'
        )
        normalized = _normalize_converter_sets(raw, CONFIG)
        self.assertIn('SET parameters.guilt = "shared"', normalized)
        self.assertIn('SET parameters.tariffs_freeze = true', normalized)
        self.assertIn('SET parameters.tax_relief = false', normalized)
        self.assertIn('SET parameters.plant_mode = "70%"', normalized)
        self.assertIn('SET parameters.timeline_months = 24', normalized)

    def test_multi_factor_scoring_with_levers_and_table_render(self):
        # 5-player config demo
        from tests.test_langflow_bridge import DEMO
        config = GameConfig.from_dict(DEMO)
        state = State()

        # Without agreed parameters
        scores_base = score_state(state, config, signed=True)

        # Apply agreed parameters: shared guilt, tariffs freeze, open sensors, plant 70%
        actions = parse_sets(
            'SET parameters.guilt = "shared"\n'
            'SET parameters.tariffs_freeze = true\n'
            'SET parameters.open_sensors = true\n'
            'SET parameters.plant_mode = "70%"',
            "city_administration",
            config,
        )
        state_agreed = merge_round(state, actions, config)
        scores_agreed = score_state(state_agreed, config, signed=True)

        # Residents score must increase with tariffs frozen and open sensors
        self.assertGreater(scores_agreed["city_residents"], scores_base["city_residents"])
        # NGO score must increase with open sensors and 70% plant mode
        self.assertGreater(scores_agreed["environmental_ngo"], scores_base["environmental_ngo"])

        # Table includes parameter rows
        table = render_table(state_agreed, config, signed=True)
        self.assertIn("Условие: Заморозка тарифов ЖКХ", table)
        self.assertIn("Условие: Режим завода", table)
        self.assertIn("Да (Администрация)", table)


if __name__ == "__main__":
    unittest.main()
