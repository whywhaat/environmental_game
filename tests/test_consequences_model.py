"""Модель последствий: итог игры → прогноз состояния города N через 10 лет."""

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from consequences_model import forecast
from consequences_model.build_component import COMPONENT_PATH, render_component
from consequences_model.build_flow import CHAT, FLOW_PATH, NODE, SOURCE_FLOW_PATH, build_flow

ROOT = Path(__file__).resolve().parents[1]

ADMIN = "city_administration"
HOLDING = "industrial_holding"
REGION = "regional_government"
RESIDENTS = "city_residents"
NGO = "environmental_ngo"
ROLES = (ADMIN, HOLDING, REGION, RESIDENTS, NGO)
MLN = 1_000_000


NAMES = {ADMIN: "Администрация", HOLDING: "Промкомплекс", REGION: "Регион", RESIDENTS: "Жители", NGO: "Эко-НКО"}
BUDGETS = {ADMIN: 30 * MLN, HOLDING: 180 * MLN, REGION: 90 * MLN, RESIDENTS: 0, NGO: 0}  # демо-flow игры
DEFAULT_CONDITIONS = {"guilt": "shared", "plant_mode": "100%", "timeline": 36, "tariffs_freeze": False,
                      "tax_relief": False, "open_sensors": False, "protest_moratorium": False, "water_supply": False}


VOTE_WORDS = {True: "ЗА", False: "ПРОТИВ", None: "ВОЗДЕРЖАЛСЯ"}
WATER_COST = 20 * MLN
MAX_DEFICIT = 100_000  # игра принимает протокол при дефиците не больше 100 тыс. ₽


def game(contributions, votes, conditions=None, accepted=None, total_cost=270 * MLN):
    """Итог игры в формате оркестратора Влада: Data.data."""
    game_conditions = {**DEFAULT_CONDITIONS, **(conditions or {})}
    votes = {role: votes.get(role, False) for role in ROLES}
    total_money = float(sum(contributions.values()))
    deficit = max(0.0, total_cost - total_money)
    if accepted is None:
        accepted = sum(vote is True for vote in votes.values()) >= 3 and deficit <= MAX_DEFICIT
    water_cost = float(WATER_COST) if game_conditions["water_supply"] else 0.0
    state_players = {NAMES[role]: {"contribution": float(contributions[role]) if role in contributions else None,
                                   "share": round(contributions[role] / total_cost * 100, 2) if role in contributions
                                   else None,
                                   "spent_on_levers": water_cost if role == ADMIN else 0.0}
                     for role in ROLES}
    return {
        "report": "", "files": {}, "history": [], "rag_retrievals": [],
        "players": [{"id": role, "name": NAMES[role], "budget": BUDGETS[role], "prompt": "Стратегия"}
                    for role in ROLES],
        "state": {"total_cost": float(total_cost), "players": state_players, "levers": game_conditions},
        "voting_summary": {"records": [{"player": NAMES[role], "vote": VOTE_WORDS[votes[role]], "comment": ""}
                                       for role in ROLES]},
        "consensus_reached": accepted,
        "total_money": total_money,
        "total_cost": float(total_cost),
        "deficit": deficit,
    }


FAIR_SPLIT = {ADMIN: 10 * MLN, HOLDING: 170 * MLN, REGION: 90 * MLN}
REFERENCE = {"open_sensors": True, "water_supply": True, "tariffs_freeze": True}
ALL_FOR = {role: True for role in ROLES}
NO_SENSORS = {"water_supply": True, "tariffs_freeze": True}
EVERY_OUTCOME = {
    "Открытая река": game(FAIR_SPLIT, ALL_FOR, REFERENCE),
    "Застой": game(FAIR_SPLIT, {}, REFERENCE),
    "Каждый сам за себя": game(FAIR_SPLIT, {ADMIN: True, HOLDING: True}, REFERENCE),
    "Недострой": game(FAIR_SPLIT, {**ALL_FOR, HOLDING: False}, NO_SENSORS),
    "Решили без жителей": game(FAIR_SPLIT, {**ALL_FOR, RESIDENTS: False, NGO: False}, REFERENCE),
    "Долгострой": game({ADMIN: 30 * MLN, HOLDING: 100 * MLN, REGION: 140 * MLN}, {**ALL_FOR, HOLDING: False}),
    "Холдинг заплатил за всех": game({ADMIN: 10 * MLN, HOLDING: 200 * MLN, REGION: 60 * MLN}, ALL_FOR, REFERENCE),
    "Город заплатил за всех": game({ADMIN: 140 * MLN, HOLDING: 80 * MLN, REGION: 50 * MLN}, ALL_FOR, REFERENCE),
    "Чистая вода, закрытые данные": game(FAIR_SPLIT, ALL_FOR, NO_SENSORS),
    "Компромисс": game(FAIR_SPLIT, {**ALL_FOR, RESIDENTS: False}, REFERENCE),
}


class OutcomeTests(unittest.TestCase):
    def test_reference_protocol_leads_to_open_river(self):
        result = forecast(game(FAIR_SPLIT, ALL_FOR, REFERENCE))
        self.assertEqual(result["outcome"]["name"], "Открытая река")
        self.assertEqual(result["commissioning_month"], 36)

    def test_rejected_protocol_without_support_leads_to_stagnation(self):
        result = forecast(game(FAIR_SPLIT, {}, REFERENCE))
        self.assertEqual(result["outcome"]["name"], "Застой")

    def test_rejected_protocol_with_some_support_leads_to_each_for_themselves(self):
        result = forecast(game(FAIR_SPLIT, {ADMIN: True, HOLDING: True}, REFERENCE))
        self.assertEqual(result["outcome"]["name"], "Каждый сам за себя")
        self.assertEqual(result["execution"]["contributions"][HOLDING], "частично")
        self.assertEqual(result["execution"]["contributions"][REGION], "нет")

    def test_main_payer_against_without_open_sensors_leads_to_unfinished_plant(self):
        result = forecast(EVERY_OUTCOME["Недострой"])
        # Холдинг вносит половину из 170 млн: покрыто 185 из 270 млн, меньше 75%.
        self.assertEqual(result["outcome"]["name"], "Недострой")
        self.assertIsNone(result["commissioning_month"])

    def test_late_plant_without_water_guarantee_leads_to_long_construction(self):
        result = forecast(EVERY_OUTCOME["Долгострой"])
        # Покрыто 220 из 270 млн: 36 мес. + 12 (реконструкция частично) + 24 (недофинансирование).
        self.assertEqual(result["commissioning_month"], 72)
        self.assertEqual(result["outcome"]["name"], "Долгострой")

    def test_protocol_without_open_sensors_leads_to_clean_water_closed_data(self):
        result = forecast(game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "open_sensors": False}))
        self.assertEqual(result["outcome"]["name"], "Чистая вода, закрытые данные")


class ExecutionTests(unittest.TestCase):
    def test_open_sensors_pull_up_reluctant_payer(self):
        result = forecast(game(FAIR_SPLIT, {**ALL_FOR, HOLDING: False}, REFERENCE))
        self.assertEqual(result["execution"]["contributions"][HOLDING], "полностью")
        self.assertEqual(result["execution"]["reconstruction"], "полностью")
        self.assertEqual(result["outcome"]["name"], "Открытая река")

    def test_reconstruction_term_comes_from_protocol(self):
        result = forecast(game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "timeline": 48}))
        self.assertEqual(result["commissioning_month"], 48)
        self.assertEqual(result["outcome"]["name"], "Открытая река")
        self.assertIn("| Реконструкция очистных | срок 48 мес. | Промышленный холдинг | полностью |", result["report"])

    def test_partly_fulfilled_condition_does_not_act(self):
        result = forecast(game(FAIR_SPLIT, {**ALL_FOR, ADMIN: False}, NO_SENSORS))
        self.assertEqual(result["execution"]["conditions"]["water_supply"], "частично")
        # Очистных через год ещё нет, а подвоз воды не действует.
        self.assertEqual(result["state"]["year_1"]["subsystems"]["ecology"], "не достигнуто")

    def test_water_guarantee_spending_counts_toward_credit(self):
        split = {ADMIN: 20 * MLN, HOLDING: 160 * MLN, REGION: 90 * MLN}
        self.assertEqual(forecast(game(split, ALL_FOR, REFERENCE))["protocol"]["credit"], [ADMIN])
        self.assertEqual(forecast(game(split, ALL_FOR, {"open_sensors": True}))["protocol"]["credit"], [])


class CityStateTests(unittest.TestCase):
    def test_reference_protocol_reaches_every_subsystem_goal_by_year_ten(self):
        state = forecast(EVERY_OUTCOME["Открытая река"])["state"]
        self.assertEqual(state["year_10"]["subsystems"], {
            "ecology": "достигнуто", "media": "достигнуто", "economy": "достигнуто",
            "law": "достигнуто", "politics": "достигнуто"})
        # Через год очистных ещё нет, но жителям возят воду.
        self.assertEqual(state["year_1"]["subsystems"]["ecology"], "частично")
        self.assertEqual(state["commissioning"]["month"], 36)

    def test_unfinished_plant_without_sensors_fails_river_law_and_politics_at_year_ten(self):
        state = forecast(EVERY_OUTCOME["Недострой"])["state"]
        self.assertIsNone(state["commissioning"]["month"])
        self.assertEqual(state["year_10"]["subsystems"]["ecology"], "не достигнуто")
        self.assertEqual(state["year_10"]["subsystems"]["law"], "не достигнуто")
        self.assertEqual(state["year_10"]["subsystems"]["politics"], "не достигнуто")

    def test_holding_on_credit_keeps_paying_but_fails_economy(self):
        result = forecast(EVERY_OUTCOME["Холдинг заплатил за всех"])  # 200 млн при бюджете 180
        self.assertEqual(result["execution"]["contributions"][HOLDING], "полностью")
        self.assertEqual(result["state"]["year_10"]["subsystems"]["economy"], "не достигнуто")


def explained(result_data):
    return [rule["text"] for rule in forecast(result_data)["explanation"]["rules"]]


class ConditionTests(unittest.TestCase):
    """Поправки от условий протокола: действуют, если условие выполнено полностью."""

    def test_plant_at_70_percent_eases_river_and_costs_jobs_until_plant_is_built(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, {"plant_mode": "70%"})
        state = forecast(result_data)["state"]
        # Без режима 70% через год: «Экология» не достигнута (ни очистных, ни воды), «Экономика» достигнута.
        self.assertEqual(state["year_1"]["subsystems"]["ecology"], "частично")
        self.assertEqual(state["year_1"]["subsystems"]["economy"], "частично")
        self.assertEqual(state["year_10"]["subsystems"]["economy"], "достигнуто")
        self.assertIn("Режим комбината 70% до ввода очистных: «Экология и наука» не ниже «частично» — сбросов "
                      "меньше; «Экономика» не выше «частично» — падают занятость и выручка.", explained(result_data))

    def test_guilt_on_holding_keeps_negative_above_target_after_first_year(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "guilt": "holding"})
        result = forecast(result_data)
        self.assertEqual(result["state"]["year_1"]["subsystems"]["media"], "частично")
        self.assertEqual(result["state"]["year_10"]["subsystems"]["media"], "достигнуто")
        self.assertFalse(result["fates"][HOLDING][0]["met"])  # негатив к холдингу не снижен до 60% за год
        self.assertIn("Вина за аварию признана за холдингом: «Массмедиа» через год не выше «частично» — негатив "
                      "к холдингу не упадёт до 60%, если вина официально на нём.", explained(result_data))

    def test_guilt_on_city_undermines_trust_in_local_power(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "guilt": "city"})
        result = forecast(result_data)
        self.assertEqual(result["state"]["year_10"]["subsystems"]["politics"], "частично")
        self.assertEqual(result["fates"][ADMIN][2], {"goal": "Вину за аварию не возложили на город", "met": False})
        self.assertIn("Вина за аварию признана за городом: «Политика» не выше «частично» — подорвано доверие к "
                      "местной власти.", explained(result_data))

    def test_tax_relief_softens_holding_credit_but_costs_region(self):
        split = {ADMIN: 10 * MLN, HOLDING: 210 * MLN, REGION: 50 * MLN}  # холдинг в кредите, область — 18,5%
        without_relief = forecast(game(split, ALL_FOR, REFERENCE))
        result_data = game(split, ALL_FOR, {**REFERENCE, "tax_relief": True})
        result = forecast(result_data)
        self.assertEqual(without_relief["state"]["year_10"]["subsystems"]["economy"], "не достигнуто")
        self.assertEqual(result["state"]["year_10"]["subsystems"]["economy"], "частично")
        self.assertTrue(without_relief["fates"][REGION][0]["met"])
        self.assertFalse(result["fates"][REGION][0]["met"])  # льготы — тоже затраты области
        self.assertIn("Налоговые льготы холдингу: его кредит не опускает «Экономику» ниже «частично».",
                      explained(result_data))

    def test_protest_moratorium_holds_only_while_works_go_on(self):
        result_data = game(FAIR_SPLIT, {**ALL_FOR, HOLDING: False}, {"protest_moratorium": True})
        result = forecast(result_data)
        self.assertIsNone(result["commissioning_month"])
        self.assertEqual(result["state"]["year_1"]["subsystems"]["politics"], "частично")
        self.assertEqual(result["state"]["year_10"]["subsystems"]["politics"], "не достигнуто")
        self.assertFalse(result["fates"][ADMIN][1]["met"])  # через 10 лет протесты мораторием не сдержать
        self.assertIn("Мораторий на протесты — мера на время работ: до ввода очистных «Политика» не ниже "
                      "«частично»; через 10 лет он не учитывается.", explained(result_data))


class FateTests(unittest.TestCase):
    def test_reference_protocol_fates_follow_role_goals(self):
        fates = forecast(EVERY_OUTCOME["Открытая река"])["fates"]
        met = {role: [check["met"] for check in checks] for role, checks in fates.items()}
        self.assertEqual(met, {
            ADMIN: [True, True, True],  # деньги дали область и холдинг; протестов нет; вина не на городе
            HOLDING: [True, True],    # негатив до 60% за год; вложения не чрезмерные
            REGION: [False, True],    # треть сметы — не «низкие затраты»; доверие восстановлено
            RESIDENTS: [True, True],  # чистая вода; тарифы заморожены на время работ
            NGO: [True, True],        # открытые датчики работают; НКО включила их в протокол
        })


class ReachabilityTests(unittest.TestCase):
    def test_every_outcome_is_reachable(self):
        for name, result_data in EVERY_OUTCOME.items():
            with self.subTest(outcome=name):
                self.assertEqual(forecast(result_data)["outcome"]["name"], name)


class ReportTests(unittest.TestCase):
    def test_report_covers_criterion_six_without_unproven_harm(self):
        for name, result_data in EVERY_OUTCOME.items():
            with self.subTest(outcome=name):
                report = forecast(result_data)["report"]
                self.assertIn(f"Исход: «{name}»", report)
                for indicator in ("Распределение затрат", "Сроки", "Экологический риск",
                                  "Недовольство и риск протестов", "Доверие к администрации", "Негатив к холдингу",
                                  "Доступность временных мер", "Рабочие места"):
                    self.assertIn(indicator, report)
                for unproven in ("болезн", "заболев", "банкрот"):
                    self.assertNotIn(unproven, report.lower())

    def test_texts_use_glossary_term_protocol(self):
        for result_data in EVERY_OUTCOME.values():
            result = forecast(result_data)
            self.assertNotIn("договор", (result["report"] + result["herald_prompt"]).lower())

    def test_report_lists_every_protocol_condition_with_owner_and_execution(self):
        report = forecast(game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "guilt": "holding"}))["report"]
        self.assertIn("| Открытые эко-датчики 24/7 | включено | Экологическая организация (НКО) | полностью |", report)
        self.assertIn("| Налоговые льготы холдингу | не включено | Региональная власть | — |", report)
        self.assertIn("| Признание вины за аварию | на холдинге | Региональная власть | решение: действует, если "
                      "протокол принят |", report)

    def test_herald_prompt_covers_three_moments_and_delay(self):
        prompt = forecast(EVERY_OUTCOME["Долгострой"])["herald_prompt"]
        self.assertIn("- Через год: Экология и наука — не достигнуто;", prompt)
        self.assertIn("- При вводе очистных (72-й мес.): Экология и наука — частично;", prompt)
        self.assertIn("- События: очистные введены на 72-м месяце, на 36 мес. позже обещанного; Региональная власть "
                      "платит в кредит.", prompt)

    def test_herald_prompt_carries_protocol_conditions(self):
        prompt = forecast(EVERY_OUTCOME["Открытая река"])["herald_prompt"]
        self.assertIn("- Условия протокола: срок реконструкции 36 мес.; режим работы комбината 100%; вина за аварию "
                      "обоюдная; включены: заморозка тарифов ЖКХ — полностью, открытые эко-датчики 24/7 — "
                      "полностью, гарантия питьевой воды — полностью.", prompt)

    def test_herald_prompt_does_not_present_conditions_of_rejected_protocol_as_acting(self):
        prompt = forecast(game(FAIR_SPLIT, {}, {**REFERENCE, "guilt": "city"}))["herald_prompt"]
        self.assertIn("- Условия протокола: протокол не принят, его условия не вступили в силу.", prompt)
        self.assertNotIn("вина за аварию", prompt)

    def test_herald_prompt_carries_outcome_and_forbids_inventing_facts(self):
        prompt = forecast(EVERY_OUTCOME["Открытая река"])["herald_prompt"]
        self.assertIn("«Вестник города N, 2036»", prompt)
        self.assertIn("Исход: «Открытая река»", prompt)
        self.assertIn("Не добавляй фактов, событий и чисел", prompt)
        self.assertIn("Учебная гипотеза", prompt)

    def test_same_game_result_always_gives_same_forecast(self):
        result_data = game(FAIR_SPLIT, {**ALL_FOR, HOLDING: False}, REFERENCE)
        self.assertEqual(forecast(result_data), forecast(result_data))


class GameResultTests(unittest.TestCase):
    def test_abstention_is_kept_and_counts_as_no_support(self):
        votes = {ADMIN: True, HOLDING: True, REGION: True, RESIDENTS: None, NGO: None}
        result = forecast(game(FAIR_SPLIT, votes, REFERENCE))
        self.assertIsNone(result["protocol"]["votes"][RESIDENTS])
        self.assertEqual(result["outcome"]["name"], "Решили без жителей")
        self.assertIn("| Жители города N | воздержался |", result["report"])

    def test_player_without_vote_record_abstains(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, REFERENCE)
        result_data["voting_summary"]["records"] = [record for record in result_data["voting_summary"]["records"]
                                                    if record["player"] != NAMES[NGO]]
        self.assertIsNone(forecast(result_data)["protocol"]["votes"][NGO])

    def test_unknown_vote_is_reported_and_counts_as_abstention(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, REFERENCE)
        result_data["voting_summary"]["records"][0]["vote"] = "НЕ ЗНАЮ"
        result = forecast(result_data)
        self.assertIsNone(result["protocol"]["votes"][ADMIN])
        self.assertEqual(result["warnings"], ["Неизвестный голос «НЕ ЗНАЮ» у игрока «Администрация»: считается "
                                              "«воздержался»."])

    def test_contributions_and_votes_are_matched_to_roles_by_player_name(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, REFERENCE)
        for profile in result_data["players"]:
            if profile["id"] == HOLDING:
                profile["name"] = "Холдинг «Север»"
        result_data["state"]["players"]["Холдинг «Север»"] = result_data["state"]["players"].pop(NAMES[HOLDING])
        for record in result_data["voting_summary"]["records"]:
            if record["player"] == NAMES[HOLDING]:
                record["player"] = "Холдинг «Север»"
        result = forecast(result_data)
        self.assertEqual(result["protocol"]["contributions"][HOLDING], 170 * MLN)
        self.assertTrue(result["protocol"]["votes"][HOLDING])
        self.assertEqual(result["warnings"], [])

    def test_unknown_player_is_reported_once_and_ignored(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, REFERENCE)
        result_data["state"]["players"]["Мэр"] = {"contribution": 10.0 * MLN, "share": 3.7, "spent_on_levers": 0.0}
        result_data["voting_summary"]["records"].append({"player": "Мэр", "vote": "ЗА", "comment": ""})
        result = forecast(result_data)
        self.assertEqual(result["warnings"], ["Неизвестный игрок «Мэр»: его взнос и голос не учитываются."])
        self.assertEqual(result["paid"], 270 * MLN)

    def test_unknown_condition_and_invalid_value_are_reported_and_defaulted(self):
        result_data = game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "timeline": 60})
        result_data["state"]["levers"]["public_council"] = True
        result = forecast(result_data)
        self.assertEqual(result["promised_months"], 36)
        self.assertEqual(result["warnings"], [
            "Недопустимое значение 60 условия «Срок реконструкции»: взято значение по умолчанию.",
            "Неизвестное условие public_council: в прогнозе не учитывается.",
        ])

    def test_missing_conditions_take_game_defaults(self):
        result_data = game(FAIR_SPLIT, ALL_FOR)
        result_data["state"]["levers"] = {}
        result = forecast(result_data)
        self.assertEqual(result["promised_months"], 36)
        self.assertEqual(result["protocol"]["included"], [])
        self.assertEqual(result["outcome"]["name"], "Чистая вода, закрытые данные")


class ExplanationTests(unittest.TestCase):
    def test_without_holding_contribution_city_would_be_left_with_unfinished_plant(self):
        without = forecast(EVERY_OUTCOME["Открытая река"])["explanation"]["without_role"]
        self.assertEqual(without[HOLDING], "Недострой")
        self.assertEqual(without[RESIDENTS], "Открытая река")

    def test_every_fired_rule_names_its_source(self):
        rules = forecast(game(FAIR_SPLIT, {**ALL_FOR, REGION: False}, NO_SENSORS))["explanation"]["rules"]
        self.assertTrue(all(rule["source"] for rule in rules))
        self.assertIn({"step": "Исполнение", "source": "Учебник, 3.1, табл. 4",
                       "text": "Открытые датчики не включены в протокол: независимого контроля нет, истинное "
                               "состояние реки неизвестно."}, rules)
        self.assertEqual(rules[-1]["text"], "Исход «Чистая вода, закрытые данные»: открытые датчики не включены "
                                            "в протокол или выполнены не полностью.")

    def test_city_state_rules_are_explained_with_sources(self):
        rules = forecast(EVERY_OUTCOME["Открытая река"])["explanation"]["rules"]
        self.assertIn({"step": "Состояние города", "source": "Учебник, 3.1, табл. 4, 5",
                       "text": "Экология и наука через 10 лет — достигнуто: очистные введены, открытые датчики "
                               "выполнены полностью."}, rules)

    def test_condition_not_fully_fulfilled_is_explained_as_not_acting(self):
        texts = explained(game(FAIR_SPLIT, {**ALL_FOR, ADMIN: False}, NO_SENSORS))
        self.assertIn("Условие «Гарантия питьевой воды» выполнено частично → на город не влияет.", texts)
        self.assertNotIn("Условие «Заморозка тарифов ЖКХ» выполнено полностью → на город не влияет.", texts)

    def test_level_set_by_condition_is_explained_by_the_condition(self):
        rules = forecast(game(FAIR_SPLIT, ALL_FOR, {**REFERENCE, "guilt": "city"}))["explanation"]["rules"]
        self.assertIn({"step": "Состояние города",
                       "source": "наша формализация по учебнику, 3.1, табл. 5 («Политика»: индекс доверия к местной "
                                 "власти)",
                       "text": "Политика через 10 лет — частично: уровень задан условием протокола, см. поправку "
                               "ниже."}, rules)

    def test_vote_rule_reads_naturally_for_rejected_protocol(self):
        texts = explained(EVERY_OUTCOME["Каждый сам за себя"])
        self.assertIn("Промышленный холдинг: голос «за», протокол не принят → заявленное выполняется частично.", texts)
        self.assertIn("Региональная власть: голос «против», протокол не принят → обязательства не выполняются.", texts)

    def test_delay_is_broken_down_in_explanation(self):
        self.assertIn("Внесено 220 из 270 млн ₽ (81%) → очистные введены на 72-м месяце: обещано 36, "
                      "+12 — реконструкция выполнена частично, +24 — смета покрыта не полностью.",
                      explained(EVERY_OUTCOME["Долгострой"]))


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "consequences_model", *args], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout

    def test_saved_game_result_prints_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "game.json"
            path.write_text(json.dumps(EVERY_OUTCOME["Открытая река"]), encoding="utf-8")
            self.assertIn("**Исход: «Открытая река».**", self.run_cli(str(path)))

    def test_json_output_carries_whole_forecast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "game.json"
            path.write_text(json.dumps(EVERY_OUTCOME["Холдинг заплатил за всех"]), encoding="utf-8")
            output = json.loads(self.run_cli(str(path), "--json"))
        self.assertEqual(output["outcome"]["name"], "Холдинг заплатил за всех")
        self.assertEqual(output["protocol"]["credit"], [HOLDING])


def load_langflow_component():
    """Загружает компонент так, как его грузит Langflow: без `from __future__` и без регистрации модуля.

    Модули lfx подменены заглушками.
    """
    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        return stub

    def fake_input(**kwargs):
        return kwargs

    stubs = {
        "lfx": module("lfx"),
        "lfx.custom": module("lfx.custom", Component=object),
        "lfx.io": module("lfx.io", DataInput=fake_input, IntInput=fake_input, Output=fake_input),
        "lfx.schema": module("lfx.schema"),
        "lfx.schema.data": module("lfx.schema.data", Data=SimpleNamespace),
        "lfx.schema.message": module("lfx.schema.message", Message=SimpleNamespace),
    }
    tree = ast.parse(COMPONENT_PATH.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
    namespace = {}
    with mock.patch.dict(sys.modules, stubs):
        exec(compile(tree, str(COMPONENT_PATH), "exec"), namespace)
    return namespace["ConsequencesModel"]


class LangflowComponentTests(unittest.TestCase):
    def test_component_file_is_built_from_current_package(self):
        self.assertEqual(COMPONENT_PATH.read_text(encoding="utf-8"), render_component(),
                         "Пересоберите компонент: python -m consequences_model.build_component")

    def test_component_imports_only_stdlib_and_langflow(self):
        # На genaiflow в JSON flow хранится только код узла: пакетов из репо там нет.
        tree = ast.parse(COMPONENT_PATH.read_text(encoding="utf-8"))
        imported = {node.module if isinstance(node, ast.ImportFrom) else alias.name
                    for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                    for alias in node.names}
        self.assertLessEqual(imported, {"__future__", "dataclasses", "json", "typing", "lfx.custom", "lfx.io",
                                        "lfx.schema.data", "lfx.schema.message"})

    def test_component_gives_same_forecast_as_package(self):
        component = load_langflow_component()()
        for result_data in EVERY_OUTCOME.values():
            component.game_result = SimpleNamespace(data=result_data)
            self.assertEqual(component.build_forecast().data, forecast(result_data))

    def test_component_reads_only_game_result_and_gives_forecast_report_and_prompt(self):
        component_class = load_langflow_component()
        self.assertEqual([item["name"] for item in component_class.inputs], ["game_result"])
        component = component_class()
        component.game_result = SimpleNamespace(data=EVERY_OUTCOME["Холдинг заплатил за всех"])
        self.assertEqual(component.build_forecast().data["protocol"]["credit"], [HOLDING])
        self.assertIn("**Исход: «Холдинг заплатил за всех».**", component.build_report().text)
        self.assertIn("«Вестник города N, 2036»", component.build_herald_prompt().text)


REBUILD_FLOW = ('Пересоберите flow: uv run --no-project --python 3.12 --with "lfx==1.12.5" '
                "python -m consequences_model.build_flow")


class LangflowFlowTests(unittest.TestCase):
    """Свой flow: граф переговоров без изменений и узел модели последствий."""

    def setUp(self):
        self.source = json.loads(SOURCE_FLOW_PATH.read_text(encoding="utf-8"))
        self.flow = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
        self.node = next(node for node in self.flow["data"]["nodes"] if node["id"] == NODE)

    def test_flow_is_built_from_current_negotiation_flow(self):
        # Шаблон узла строит lfx, всё остальное берётся из графа переговоров.
        self.assertEqual(self.flow, build_flow(self.source, self.node["data"]["node"]), REBUILD_FLOW)

    def test_flow_node_carries_current_component_code(self):
        self.assertEqual(self.node["data"]["node"]["template"]["code"]["value"],
                         COMPONENT_PATH.read_text(encoding="utf-8"), REBUILD_FLOW)

    def test_negotiation_graph_is_kept_unchanged(self):
        nodes = {node["id"]: node for node in self.flow["data"]["nodes"]}
        for node in self.source["data"]["nodes"]:
            self.assertEqual(nodes[node["id"]], node)
        for edge in self.source["data"]["edges"]:
            self.assertIn(edge, self.flow["data"]["edges"])
        self.assertNotEqual(self.flow.get("id"), self.source.get("id"), "импорт перезаписал бы flow игры")

    def test_model_reads_game_result_and_reports_to_its_chat(self):
        links = {(edge["source"], edge["target"], edge["data"]["targetHandle"]["fieldName"])
                 for edge in self.flow["data"]["edges"]}
        added = links - {(edge["source"], edge["target"], edge["data"]["targetHandle"]["fieldName"])
                         for edge in self.source["data"]["edges"]}
        self.assertEqual(added, {("TwoLLMContractGame-main", NODE, "game_result"), (NODE, CHAT, "input_value")})


class ComponentBuildTests(unittest.TestCase):
    """Сборка отказывается собирать файл, который вышел бы молча неправильным."""

    def build_with(self, module, extra_code):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / "consequences_model"
            shutil.copytree(ROOT / "consequences_model", package, ignore=shutil.ignore_patterns("__pycache__"))
            path = package / f"{module}.py"
            source = path.read_text(encoding="utf-8") if path.exists() else '"""Новый модуль."""\n'
            path.write_text(source + extra_code, encoding="utf-8")
            return render_component(package)

    def test_module_missing_from_build_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "extra"):
            self.build_with("extra", "\nEXTRA = 1\n")

    def test_import_from_module_built_later_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "forecast"):
            self.build_with("catalog", "\nfrom .model import forecast\n")

    def test_definition_shadowing_shared_import_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "field"):
            self.build_with("state", "\n\ndef field():\n    return None\n")


def play_orchestrator_game(replies, votes):
    """Партия настоящим оркестратором игры Влада (dialog.py): модули Langflow подменены заглушками.

    replies — что каждый игрок говорит на всех этапах до голосования, votes — его ответ на голосовании.
    """
    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        return stub

    def fake_input(**kwargs):
        return kwargs

    def message_class(name, kind):
        return type(name, (), {"type": kind, "__init__": lambda self, content: setattr(self, "content", content)})

    stubs = {
        "lfx": module("lfx"),
        "lfx.custom": module("lfx.custom", Component=object),
        "lfx.io": module("lfx.io", DataInput=fake_input, HandleInput=fake_input, IntInput=fake_input,
                         MultilineInput=fake_input, Output=fake_input),
        "lfx.schema": module("lfx.schema"),
        "lfx.schema.data": module("lfx.schema.data", Data=SimpleNamespace),
        "lfx.schema.message": module("lfx.schema.message", Message=SimpleNamespace),
        "langchain_core": module("langchain_core"),
        "langchain_core.messages": module("langchain_core.messages", AIMessage=message_class("AIMessage", "ai"),
                                          HumanMessage=message_class("HumanMessage", "human"),
                                          SystemMessage=message_class("SystemMessage", "system")),
    }
    path = ROOT / "langflow/custom_components/my_components/dialog.py"
    namespace = {}
    with mock.patch.dict(sys.modules, stubs):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)

    class ScriptedModel:
        def __init__(self, role):
            self.role = role

        def invoke(self, messages):
            voting = "ЭТАП 3" in messages[0].content
            return SimpleNamespace(content=votes[self.role] if voting else replies[self.role])

    game_node = namespace["TwoLLMContractGame"]()
    game_node.players = [SimpleNamespace(data={"id": role, "name": NAMES[role], "budget": BUDGETS[role],
                                               "prompt": "Стратегия", "llm": ScriptedModel(role)}) for role in ROLES]
    game_node.rag_config, game_node.use_rag, game_node.settings = None, False, None
    game_node.turns, game_node.rules_prompt, game_node.debug = 5, "", False
    return game_node.run_chat().data


class OrchestratorContractTests(unittest.TestCase):
    """Итог настоящего оркестратора игры Влада читается моделью последствий."""

    def test_game_played_by_orchestrator_is_forecast(self):
        replies = {
            ADMIN: "SPEAK Город берёт воду и тарифы.\nSET взнос 10000000\nSET water_supply да\nSET tariffs_freeze да",
            HOLDING: "SPEAK Холдинг платит основную часть.\nSET взнос 170000000",
            REGION: "SPEAK Область добавит.\nSET взнос 90000000",
            RESIDENTS: "SPEAK Нам нужна чистая вода.",
            NGO: "SPEAK Ставим открытые датчики.\nSET open_sensors да",
        }
        votes = {role: "VOTE ЗА Поддерживаем." for role in ROLES}
        votes[RESIDENTS] = "VOTE ВОЗДЕРЖАЛСЯ Подождём, что будет с водой."

        game_result = play_orchestrator_game(replies, votes)
        self.assertTrue(game_result["consensus_reached"])
        result = forecast(game_result)
        self.assertEqual(result["warnings"], [])
        self.assertIsNone(result["protocol"]["votes"][RESIDENTS])
        self.assertEqual(result["protocol"]["included"], ["tariffs_freeze", "open_sensors", "water_supply"])
        self.assertEqual(result["protocol"]["credit"], [])  # 10 млн взноса + 20 млн на воду = бюджет администрации
        self.assertEqual(result["outcome"]["name"], "Компромисс")


if __name__ == "__main__":
    unittest.main()
