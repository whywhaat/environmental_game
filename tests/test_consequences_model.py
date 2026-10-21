"""Модель последствий: итог игры → прогноз состояния города N через 10 лет."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from consequences_model import forecast

ROOT = Path(__file__).resolve().parents[1]

ADMIN = "city_administration"
HOLDING = "industrial_holding"
REGION = "regional_government"
RESIDENTS = "city_residents"
NGO = "environmental_ngo"
ROLES = (ADMIN, HOLDING, REGION, RESIDENTS, NGO)
MLN = 1_000_000


def game(contributions, responsible, votes, accepted=None, total_cost=270 * MLN):
    """Итог игры в формате ядра переговоров: Data.data оркестратора."""
    fields = {}
    for role, amount in contributions.items():
        fields[f"participants.{role}.contribution"] = [{"value": amount, "player": role, "seq": 1}]
    for component, role in responsible.items():
        fields[f"components.{component}.responsible"] = [{"value": role, "player": role, "seq": 2}]
    votes = {role: votes.get(role, False) for role in ROLES}
    if accepted is None:
        accepted = sum(votes.values()) > len(ROLES) / 2 and sum(contributions.values()) >= total_cost
    return {"result": {"total_cost": total_cost, "final_state": {"fields": fields},
                       "votes": votes, "accepted": accepted, "final_scores": {}}}


FAIR_SPLIT = {ADMIN: 40 * MLN, HOLDING: 150 * MLN, REGION: 80 * MLN}
INDEPENDENT = {"water_supply": ADMIN, "independent_monitoring": REGION, "reconstruction": HOLDING}
ALL_FOR = {role: True for role in ROLES}


class OutcomeTests(unittest.TestCase):
    def test_reference_deal_leads_to_open_river(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))
        self.assertEqual(result["outcome"]["name"], "Открытая река")

    def test_rejected_deal_with_no_support_leads_to_stagnation(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, {}))
        self.assertEqual(result["outcome"]["name"], "Застой")

    def test_rejected_deal_with_some_support_leads_to_each_for_themselves(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, {ADMIN: True, HOLDING: True}))
        self.assertEqual(result["outcome"]["name"], "Каждый сам за себя")
        self.assertEqual(result["execution"]["contributions"][HOLDING], "частично")
        self.assertEqual(result["execution"]["contributions"][REGION], "нет")

    def test_main_payer_against_without_independent_control_leads_to_unfinished_plant(self):
        self_monitored = {**INDEPENDENT, "independent_monitoring": HOLDING, "reconstruction": REGION}
        result = forecast(game(FAIR_SPLIT, self_monitored, {**ALL_FOR, HOLDING: False}))
        # Холдинг вносит половину из 150 млн: покрыто 195 из 270 млн, меньше 75%.
        self.assertEqual(result["outcome"]["name"], "Недострой")
        self.assertIsNone(result["commissioning_month"])

    def test_independent_monitoring_pulls_up_reluctant_payer(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, HOLDING: False}))
        self.assertEqual(result["execution"]["contributions"][HOLDING], "полностью")
        self.assertEqual(result["commissioning_month"], 36)
        self.assertEqual(result["outcome"]["name"], "Открытая река")

    def test_deal_passed_over_residents_and_ngo_leads_to_decided_without_residents(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, RESIDENTS: False, NGO: False}))
        self.assertEqual(result["outcome"]["name"], "Решили без жителей")

    def test_late_plant_without_water_for_residents_leads_to_long_construction(self):
        split = {ADMIN: 60 * MLN, HOLDING: 90 * MLN, REGION: 120 * MLN}
        holding_runs_all = {"water_supply": HOLDING, "independent_monitoring": HOLDING, "reconstruction": HOLDING}
        result = forecast(game(split, holding_runs_all, {**ALL_FOR, HOLDING: False}))
        # Покрыто 225 из 270 млн: 36 мес. + 12 (реконструкция частично) + 24 (недофинансирование).
        self.assertEqual(result["commissioning_month"], 72)
        self.assertEqual(result["outcome"]["name"], "Долгострой")

    def test_holding_carrying_most_of_the_cost_leads_to_holding_paid_for_all(self):
        split = {ADMIN: 30 * MLN, HOLDING: 200 * MLN, REGION: 40 * MLN}
        result = forecast(game(split, INDEPENDENT, ALL_FOR))
        self.assertEqual(result["outcome"]["name"], "Холдинг заплатил за всех")

    def test_city_carrying_half_of_the_cost_leads_to_city_paid_for_all(self):
        split = {ADMIN: 140 * MLN, HOLDING: 80 * MLN, REGION: 50 * MLN}
        result = forecast(game(split, INDEPENDENT, ALL_FOR))
        self.assertEqual(result["outcome"]["name"], "Город заплатил за всех")

    def test_holding_monitoring_itself_leads_to_clean_water_closed_data(self):
        own_lab = {**INDEPENDENT, "independent_monitoring": HOLDING}
        result = forecast(game(FAIR_SPLIT, own_lab, ALL_FOR))
        self.assertEqual(result["outcome"]["name"], "Чистая вода, закрытые данные")

    def test_good_deal_without_residents_support_leads_to_compromise(self):
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, RESIDENTS: False}))
        self.assertEqual(result["outcome"]["name"], "Компромисс")


class ExecutionTests(unittest.TestCase):
    def test_residents_cannot_fully_run_reconstruction(self):
        result = forecast(game(FAIR_SPLIT, {**INDEPENDENT, "reconstruction": RESIDENTS}, ALL_FOR))
        self.assertEqual(result["execution"]["components"]["reconstruction"], "частично")
        self.assertEqual(result["commissioning_month"], 48)

    def test_ngo_can_fully_run_monitoring(self):
        result = forecast(game(FAIR_SPLIT, {**INDEPENDENT, "independent_monitoring": NGO}, ALL_FOR))
        self.assertEqual(result["execution"]["components"]["independent_monitoring"], "полностью")

    def test_component_without_responsible_is_not_done(self):
        result = forecast(game(FAIR_SPLIT, {"water_supply": ADMIN, "independent_monitoring": REGION}, ALL_FOR))
        self.assertEqual(result["execution"]["components"]["reconstruction"], "нет")
        self.assertEqual(result["outcome"]["name"], "Недострой")


class CityStateTests(unittest.TestCase):
    def test_reference_deal_reaches_every_subsystem_goal_by_year_ten(self):
        state = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))["state"]
        self.assertEqual(state["year_10"]["subsystems"], {
            "ecology": "достигнуто", "media": "достигнуто", "economy": "достигнуто",
            "law": "достигнуто", "politics": "достигнуто"})
        # Через год очистных ещё нет, но жителям возят воду.
        self.assertEqual(state["year_1"]["subsystems"]["ecology"], "частично")
        self.assertEqual(state["commissioning"]["month"], 36)

    def test_unfinished_plant_leaves_river_and_politics_failed_at_year_ten(self):
        no_reconstruction = {"water_supply": ADMIN, "independent_monitoring": REGION}
        state = forecast(game(FAIR_SPLIT, no_reconstruction, ALL_FOR))["state"]
        self.assertIsNone(state["commissioning"]["month"])
        self.assertEqual(state["year_10"]["subsystems"]["ecology"], "не достигнуто")
        self.assertEqual(state["year_10"]["subsystems"]["politics"], "не достигнуто")
        self.assertEqual(state["year_10"]["subsystems"]["law"], "частично")

    def test_holding_on_credit_keeps_paying_but_fails_economy(self):
        players = [{"id": ADMIN, "budget": 10 * MLN}, {"id": HOLDING, "budget": 50 * MLN},
                   {"id": REGION, "budget": 100 * MLN}, {"id": RESIDENTS, "budget": 0}, {"id": NGO, "budget": 0}]
        result = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR), players=players)
        self.assertEqual(result["execution"]["contributions"][HOLDING], "полностью")
        self.assertEqual(result["state"]["year_10"]["subsystems"]["economy"], "не достигнуто")


class FateTests(unittest.TestCase):
    def test_reference_deal_fates_follow_role_goals(self):
        fates = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))["fates"]
        met = {role: [check["met"] for check in checks] for role, checks in fates.items()}
        self.assertEqual(met, {
            ADMIN: [True, True],      # деньги дали область и холдинг; протестов нет
            HOLDING: [True, True],    # негатив до 60% за год; вложения не чрезмерные
            REGION: [False, True],    # 30% сметы — не «низкие затраты»; доверие восстановлено
            RESIDENTS: [True, True],  # чистая вода; тарифы заморожены на время работ
            NGO: [True, False],       # независимый контроль есть; мониторинг ведёт не НКО
        })


EVERY_OUTCOME = {
    "Открытая река": game(FAIR_SPLIT, INDEPENDENT, ALL_FOR),
    "Застой": game(FAIR_SPLIT, INDEPENDENT, {}),
    "Каждый сам за себя": game(FAIR_SPLIT, INDEPENDENT, {ADMIN: True, HOLDING: True}),
    "Недострой": game(FAIR_SPLIT, {"water_supply": ADMIN, "independent_monitoring": REGION}, ALL_FOR),
    "Решили без жителей": game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, RESIDENTS: False, NGO: False}),
    "Долгострой": game({ADMIN: 60 * MLN, HOLDING: 90 * MLN, REGION: 120 * MLN},
                       {"water_supply": HOLDING, "independent_monitoring": HOLDING, "reconstruction": HOLDING},
                       {**ALL_FOR, HOLDING: False}),
    "Холдинг заплатил за всех": game({ADMIN: 30 * MLN, HOLDING: 200 * MLN, REGION: 40 * MLN}, INDEPENDENT, ALL_FOR),
    "Город заплатил за всех": game({ADMIN: 140 * MLN, HOLDING: 80 * MLN, REGION: 50 * MLN}, INDEPENDENT, ALL_FOR),
    "Чистая вода, закрытые данные": game(FAIR_SPLIT, {**INDEPENDENT, "independent_monitoring": HOLDING}, ALL_FOR),
    "Компромисс": game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, RESIDENTS: False}),
}


class ReportTests(unittest.TestCase):
    def test_every_outcome_is_reachable(self):
        for name, result_data in EVERY_OUTCOME.items():
            self.assertEqual(forecast(result_data)["outcome"]["name"], name)

    def test_report_covers_criterion_six_without_unproven_harm(self):
        for name, result_data in EVERY_OUTCOME.items():
            report = forecast(result_data)["report"]
            self.assertIn(f"Исход: «{name}»", report)
            for indicator in ("Распределение затрат", "Сроки", "Экологический риск", "Недовольство и риск протестов",
                              "Доверие к администрации", "Негатив к холдингу", "Доступность временных мер",
                              "Рабочие места"):
                self.assertIn(indicator, report)
            for unproven in ("болезн", "заболев", "банкрот"):
                self.assertNotIn(unproven, report.lower())

    def test_herald_prompt_carries_outcome_and_forbids_inventing_facts(self):
        prompt = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))["herald_prompt"]
        self.assertIn("«Вестник города N, 2036»", prompt)
        self.assertIn("Исход: «Открытая река»", prompt)
        self.assertIn("Не добавляй фактов, событий и чисел", prompt)
        self.assertIn("Учебная гипотеза", prompt)

    def test_same_game_result_always_gives_same_forecast(self):
        result_data = game(FAIR_SPLIT, INDEPENDENT, {**ALL_FOR, HOLDING: False})
        self.assertEqual(forecast(result_data), forecast(result_data))


class GameResultTests(unittest.TestCase):
    def test_saved_result_json_without_langflow_wrapper_gives_same_forecast(self):
        wrapped = game(FAIR_SPLIT, INDEPENDENT, ALL_FOR)
        self.assertEqual(forecast(wrapped["result"]), forecast(wrapped))

    def test_conflicting_contribution_counts_as_undeclared(self):
        result_data = game(FAIR_SPLIT, INDEPENDENT, ALL_FOR, accepted=False)
        result_data["result"]["final_state"]["fields"]["participants.industrial_holding.contribution"].append(
            {"value": 100 * MLN, "player": REGION, "seq": 3})
        result = forecast(result_data)
        self.assertNotIn(HOLDING, result["execution"]["contributions"])
        self.assertIn("Конфликт в поле participants.industrial_holding.contribution: обязательство считается "
                      "незаявленным.", result["warnings"])

    def test_unknown_component_and_role_are_reported_and_ignored(self):
        result_data = game(FAIR_SPLIT, {**INDEPENDENT, "water_supply": "mayor", "extra_works": ADMIN}, ALL_FOR)
        result = forecast(result_data)
        self.assertEqual(result["execution"]["components"]["water_supply"], "нет")
        self.assertEqual(result["warnings"], [
            "Неизвестная роль mayor в поле components.water_supply.responsible: компонент без ответственного.",
            "Неизвестный компонент extra_works: в прогнозе не учитывается.",
        ])


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "consequences_model", *args], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout

    def test_saved_game_prints_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            path.write_text(json.dumps(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR)["result"]), encoding="utf-8")
            self.assertIn("**Исход: «Открытая река».**", self.run_cli(str(path)))

    def test_game_config_supplies_budgets_and_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path, config_path = Path(tmp) / "result.json", Path(tmp) / "config.json"
            result_path.write_text(json.dumps(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR)["result"]), encoding="utf-8")
            config_path.write_text(json.dumps({"players": [{"id": HOLDING, "budget": 50 * MLN}]}), encoding="utf-8")
            output = json.loads(self.run_cli(str(result_path), "--config", str(config_path), "--months", "48", "--json"))
        self.assertEqual(output["protocol"]["credit"], [HOLDING])
        self.assertEqual(output["commissioning_month"], 48)


def load_langflow_component():
    """Загружает компонент без Langflow: модули lfx подменены заглушками."""
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
    path = ROOT / "langflow/custom_components/my_components/ConsequencesModel.py"
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("consequences_component", path)
        component_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(component_module)
    return component_module.ConsequencesModel


class LangflowComponentTests(unittest.TestCase):
    def test_component_turns_orchestrator_output_into_forecast_report_and_prompt(self):
        component = load_langflow_component()()
        component.game_result = SimpleNamespace(data=game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))
        component.players = [SimpleNamespace(data={"id": HOLDING, "budget": 50 * MLN, "llm": object()})]
        component.promised_months = 36

        self.assertEqual(component.build_forecast().data["outcome"]["name"], "Открытая река")
        self.assertEqual(component.build_forecast().data["protocol"]["credit"], [HOLDING])
        self.assertIn("**Исход: «Открытая река».**", component.build_report().text)
        self.assertIn("«Вестник города N, 2036»", component.build_herald_prompt().text)


class ExplanationTests(unittest.TestCase):
    def test_without_holding_contribution_city_would_be_left_with_unfinished_plant(self):
        without = forecast(game(FAIR_SPLIT, INDEPENDENT, ALL_FOR))["explanation"]["without_role"]
        self.assertEqual(without[HOLDING], "Недострой")
        self.assertEqual(without[RESIDENTS], "Открытая река")

    def test_every_fired_rule_names_its_source(self):
        own_lab = {**INDEPENDENT, "independent_monitoring": HOLDING}
        rules = forecast(game(FAIR_SPLIT, own_lab, {**ALL_FOR, REGION: False}))["explanation"]["rules"]
        self.assertTrue(all(rule["source"] for rule in rules))
        self.assertIn({"step": "Исполнение", "source": "Учебник, 3.1, табл. 4",
                       "text": "Мониторинг ведёт Промышленный холдинг — «своя лаборатория»: "
                               "истинное состояние реки неизвестно."}, rules)
        self.assertEqual(rules[-1]["text"], "Исход «Чистая вода, закрытые данные»: "
                                            "мониторинг не независимый или выполнен не полностью.")

    def test_delay_is_broken_down_in_explanation(self):
        texts = [rule["text"] for rule in forecast(EVERY_OUTCOME["Долгострой"])["explanation"]["rules"]]
        self.assertIn("Внесено 225 из 270 млн ₽ (83%) → очистные введены на 72-м месяце: обещано 36, "
                      "+12 — реконструкция выполнена частично, +24 — смета покрыта не полностью.", texts)


if __name__ == "__main__":
    unittest.main()
