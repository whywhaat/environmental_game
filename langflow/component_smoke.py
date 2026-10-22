"""Exercise the actual custom-component boundaries with deterministic models."""

import json
from pathlib import Path

from my_components.PlayerModel import PlayerAgent
from my_components.dialog import TwoLLMContractGame
from my_components.game_settings import GameSettings
from my_components.report_controller import ReportController


class DummyModel:
    def invoke(self, messages):
        contents = [str(item.content) for item in messages]
        if contents[0].startswith("Ты конвертер"):
            answer = "NO_CHANGES"
        elif any("Голосование по договору открыто" in content for content in contents):
            answer = "VOTE ПРОТИВ"
        else:
            answer = "Без изменений."
        return type("Response", (), {"content": answer})()


def main():
    config = json.loads(Path("/app/negotiation_flow/demo_config.json").read_text(encoding="utf-8"))
    settings = GameSettings()
    settings.water_cost = 50
    settings.monitoring_cost = 20
    settings.reconstruction_cost = 200
    settings.max_rounds = 1
    players = []
    for item in config["players"]:
        player = PlayerAgent()
        player.player_id = item["id"]
        player.player_name = item["name"]
        player.budget_millions = item["budget"] // 1_000_000
        player.expected_own_share = item["expected_own_share"]
        player.desired_responsible = json.dumps(item["desired_responsible"], ensure_ascii=False)
        player.personal_context = item["personal_context"]
        player.motivation = item["motivation"]
        player.personal_history = json.dumps(item["personal_history"], ensure_ascii=False)
        player.model = DummyModel()
        players.append(player.build_player())
    game = TwoLLMContractGame()
    game.settings = settings.build_settings()
    game.players = players
    game.converter_model = DummyModel()
    output = game.run_chat()
    report = ReportController()
    report.game_result = output
    message = report.generate_report()
    summary = {"rounds": len(output.data["result"]["rounds"]), "votes": len(output.data["result"]["votes"]),
               "files": output.data["files"], "report_has_paths": "Файлы запуска" in message.text}
    Path("/app/games/component_smoke.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
