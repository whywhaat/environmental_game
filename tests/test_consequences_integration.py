"""Публичный итог нового structured-ядра читается моделью последствий."""

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from consequences_model import forecast
from test_structured_turns import Data, Game, IDS, Model, NAMES, wire_turn


class AgreedContributionModel(Model):
    """Внешний LLM выдаёт заданный допустимый tool call на каждой фазе."""

    def __init__(self, millions):
        super().__init__()
        self.millions = millions

    def invoke(self, messages):
        voting = any("ИДЕТ ЭТАП 3:" in message.content for message in messages)
        self.replies.append(wire_turn(
            own_contribution=dict(action="unchanged" if voting else "set",
                                  amount_millions=0 if voting else self.millions),
            vote="for" if voting else "none"))
        return super().invoke(messages)


class ConsequencesIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_typed_game_preserves_rubles_and_acceptance_with_a_budget_gap(self):
        for millions, paid, month in ((54, 270_000_000, 36), (52, 260_000_000, 60)):
            with self.subTest(millions=millions), tempfile.TemporaryDirectory() as directory:
                game = Game()
                game.players = [Data(dict(id=role, name=name, budget=200_000_000,
                                          prompt="Проверка контракта.", llm=AgreedContributionModel(millions)))
                                for role, name in zip(IDS, NAMES)]
                game.turns, game.settings, game.rules_prompt = 5, None, ""
                game.use_rag = False
                with patch.dict(os.environ, {"NEGOTIATION_GAMES_DIR": directory}), patch("asyncio.sleep", new=AsyncMock()):
                    raw = (await game.run_chat()).data
                self.assertEqual(raw["status"], "completed")
                self.assertTrue(raw["consensus_reached"])
                self.assertEqual(raw["total_money"], paid)
                result = forecast(raw)
                self.assertTrue(result["protocol"]["accepted"])
                self.assertEqual(result["paid"], paid)
                self.assertEqual(result["commissioning_month"], month)
                self.assertEqual(result["warnings"], [])
                self.assertEqual(len(result["herald_text"].splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
