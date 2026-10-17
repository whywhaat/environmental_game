"""Unit tests for the Calculator, Draft Treaty, Voter Checklist, and Last Call logic."""

import sys
from types import ModuleType, SimpleNamespace
import unittest

# Mock lfx before importing dialog
if "lfx" not in sys.modules:
    lfx_mod = ModuleType("lfx")
    lfx_custom = ModuleType("lfx.custom")
    lfx_custom.Component = object
    lfx_inputs = ModuleType("lfx.inputs")
    lfx_inputs_inputs = ModuleType("lfx.inputs.inputs")
    fake = lambda **kwargs: kwargs
    lfx_inputs_inputs.DataInput = fake
    lfx_inputs_inputs.IntInput = fake
    lfx_io = ModuleType("lfx.io")
    lfx_io.Output = fake
    lfx_io.MultilineInput = fake
    lfx_io.DataInput = fake
    lfx_io.IntInput = fake
    lfx_schema = ModuleType("lfx.schema")
    lfx_schema.__path__ = []
    lfx_schema_data = ModuleType("lfx.schema.data")
    lfx_schema_data.Data = SimpleNamespace
    lfx_schema_msg = ModuleType("lfx.schema.message")
    lfx_schema_msg.Message = SimpleNamespace

    sys.modules["lfx"] = lfx_mod
    sys.modules["lfx.custom"] = lfx_custom
    sys.modules["lfx.inputs"] = lfx_inputs
    sys.modules["lfx.inputs.inputs"] = lfx_inputs_inputs
    sys.modules["lfx.io"] = lfx_io
    sys.modules["lfx.schema"] = lfx_schema
    sys.modules["lfx.schema.data"] = lfx_schema_data
    sys.modules["lfx.schema.message"] = lfx_schema_msg

if "langchain_core" not in sys.modules:
    lc_mod = ModuleType("langchain_core")
    lc_msg = ModuleType("langchain_core.messages")
    lc_msg.HumanMessage = SimpleNamespace
    lc_msg.SystemMessage = SimpleNamespace
    sys.modules["langchain_core"] = lc_mod
    sys.modules["langchain_core.messages"] = lc_msg

from langflow.custom_components.my_components.dialog import (
    TwoLLMContractGame,
    _clean_speech_message,
    _format_money,
)


class CalculatorAndTreatyTests(unittest.TestCase):
    def setUp(self):
        self.levers_config = [
            {
                "id": "timeline_months",
                "title": "Срок реконструкции (мес.)",
                "type": "int",
                "owner": "Промышленный холдинг",
                "cost": 0.0,
                "default": 36,
            },
            {
                "id": "water_supply_guarantee",
                "title": "Гарантия питьевой воды жителям",
                "type": "bool",
                "owner": "Администрация города N",
                "cost": 50_000_000.0,
                "default": False,
                "description": "Поставка чистой питьевой воды жителям",
            },
            {
                "id": "independent_monitoring",
                "title": "Независимый экологический мониторинг",
                "type": "bool",
                "owner": "Региональная власть",
                "cost": 20_000_000.0,
                "default": False,
                "description": "Привлечение независимых лабораторий",
            },
            {
                "id": "public_council",
                "title": "Общественный контроль и комиссия",
                "type": "bool",
                "owner": "Экологическая организация (НКО)",
                "cost": 0.0,
                "default": False,
                "description": "Включение жителей и НКО в совет проекта",
            },
        ]
        self.total_cost = 1_000_000_000.0
        self.state = {
            "players": {
                "Промышленный холдинг": {"contribution": 600_000_000.0, "share": 60.0, "spent_on_levers": 0.0},
                "Региональная власть": {"contribution": 250_000_000.0, "share": 25.0, "spent_on_levers": 20_000_000.0},
                "Администрация города N": {"contribution": 30_000_000.0, "share": 3.0, "spent_on_levers": 50_000_000.0},
                "Экологическая организация (НКО)": {"contribution": 10_000_000.0, "share": 1.0, "spent_on_levers": 0.0},
                "Жители города N": {"contribution": None, "share": None, "spent_on_levers": 0.0},
            },
            "levers": {
                "timeline_months": 24,
                "water_supply_guarantee": True,
                "independent_monitoring": True,
                "public_council": True,
            },
        }

    def test_clean_speech_message(self):
        dirty = "SPEAK Добрый день! Мы согласны. SPEAK А также требуем гарантий. TO @Региональная власть"
        cleaned = _clean_speech_message(dirty)
        self.assertNotIn("SPEAK", cleaned)
        self.assertNotIn("TO @", cleaned)
        self.assertIn("Добрый день! Мы согласны. А также требуем гарантий.", cleaned)

    def test_calculator_renders_exact_deficit(self):
        calc = TwoLLMContractGame._render_calculator(self.state, self.total_cost)
        # Total is 600 + 250 + 30 + 10 = 890M. Deficit is 110M.
        self.assertIn("110 млн руб.", calc)
        self.assertIn("890 млн руб.", calc)
        self.assertIn("1000 млн руб.", calc)
        self.assertIn("ОФИЦИАЛЬНЫЙ КАЛЬКУЛЯТОР БЮДЖЕТА ПРОЕКТА", calc)
        self.assertIn("НЕ СЧИТАЙ В УМЕ", calc)

    def test_draft_treaty_renders_all_sections(self):
        treaty = TwoLLMContractGame._render_draft_treaty(self.state, self.levers_config, self.total_cost)
        self.assertNotIn("===", treaty)
        self.assertIn("Официальный проект соглашения", treaty)
        self.assertIn("1. Финансовый план реконструкции очистных сооружений", treaty)
        self.assertIn("2. Пакет обязательств и условий реализации", treaty)
        self.assertIn("ДЕФИЦИТ 110 млн руб.", treaty)
        self.assertIn("✅ **[УТВЕРЖДЕНО] Гарантия питьевой воды жителям**", treaty)
        self.assertIn("⏱️ **[УТВЕРЖДЕНО] Срок реконструкции (мес.):** 24 мес.", treaty)

    def test_voter_checklist_for_residents(self):
        profile = {
            "money": 0,
            "attrs": {
                "min_share": 0,
                "red_lines": ["Не допустить роста тарифов ЖКХ"],
            },
        }
        checklist = TwoLLMContractGame._render_voter_checklist(
            "Жители города N", profile, self.state, self.levers_config, self.total_cost
        )
        self.assertIn("ПЕРСОНАЛЬНЫЙ ЧЕК-ЛИСТ ДЛЯ СТОРОНЫ: @Жители города N", checklist)
        self.assertIn("Гарантия питьевой воды жителям: УТВЕРЖДЕНО ✅", checklist)
        self.assertIn("Независимый экологический мониторинг: УТВЕРЖДЕНО ✅", checklist)
        self.assertIn("Срок реконструкции (мес.): УТВЕРЖДЕНО (24 мес.) ✅", checklist)
        self.assertIn("дефицит 110 млн руб.", checklist)


if __name__ == "__main__":
    unittest.main()
