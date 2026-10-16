"""Protocol smoke tests without the Langflow runtime or network models."""

import ast
import copy
import json
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from config.prompts import judge_event, read_prompt, render_prompt
from config.prompts.prompt_config import PROMPTS
from config.prompts.judje_prompt_config import PROMPTS as JUDGE_PROMPTS
from config.prompts.player_prompt_config import PROMPTS as PLAYER_PROMPTS
from config.prompts.role_holding_prompt_config import PROMPTS as HOLDING_PROMPTS


ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class Data:
    def __init__(self, text=None, data=None):
        self.text = text
        self.data = data


def fake_input(**_kwargs):
    return None


judge_namespace = {
    "json": json,
    "Component": object,
    "Data": Data,
    "ModelInput": fake_input,
    "MultilineInput": fake_input,
    "Output": fake_input,
    "SystemMessage": SimpleNamespace,
    "HumanMessage": SimpleNamespace,
    "_extract_text": lambda content: content.strip(),
    "PROMPT_LOGGER": SimpleNamespace(warning=lambda *_args: None),
    "judge_event": judge_event,
    "read_prompt": read_prompt,
    "render_prompt": render_prompt,
}
MeetingJudgeService = load_class(
    ROOT / "langflow/custom_components/my_components/MeetingJudge.py",
    "MeetingJudgeService", judge_namespace,
)

dialog_source = ROOT / "langflow/custom_components/my_components/dialog.py"
dialog_tree = ast.parse(dialog_source.read_text(encoding="utf-8"))
dialog_namespace = {
    "copy": copy,
    "json": json,
    "re": re,
    "Component": object,
    "Data": Data,
    "BoolInput": fake_input,
    "DataInput": fake_input,
    "MultilineInput": fake_input,
    "Output": fake_input,
    "SystemMessage": SimpleNamespace,
    "HumanMessage": SimpleNamespace,
    "TOTAL_PROJECT_COST": 1_000_000_000.0,
    "PROMPT_LOGGER": SimpleNamespace(warning=lambda *_args: None),
    "judge_event": judge_event,
    "read_prompt": read_prompt,
    "render_prompt": render_prompt,
}
helpers = [item for item in dialog_tree.body if isinstance(item, ast.FunctionDef)]
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(dialog_source), "exec"), dialog_namespace)
TwoLLMContractGame = load_class(dialog_source, "TwoLLMContractGame", dialog_namespace)


class PlayerModel:
    def __init__(self, name, pledges, malformed_round=None, semicolon_pledge=False):
        self.name = name
        self.pledges = pledges
        self.malformed_round = malformed_round
        self.semicolon_pledge = semicolon_pledge
        self.prompts = []

    def invoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        if "Подготовь для " in prompt and "краткую записку" in prompt:
            return SimpleNamespace(content=(
                "Сейчас: отвечай на поручение председателя.\n"
                "Не уступай без причины: уточни заявление другой стороны.\n"
                "Последствия: соблюдай свой бюджет и регламент."
            ))
        kind = re.search(r"ТВОЁ ДЕЙСТВИЕ: (\w+)", prompt).group(1)
        round_number = int(re.search(r"КРУГ (\d+)", prompt).group(1))
        if kind == "PLEDGE":
            if round_number == self.malformed_round:
                return SimpleNamespace(content="SET me Взнос 500")
            separator = "; " if self.semicolon_pledge else "\n"
            return SimpleNamespace(content=(
                f"SET me Взнос {self.pledges[round_number - 1]}{separator}"
                "SPEAK Предлагаю этот взнос с учетом бюджета и экологических обязанностей моей стороны."
            ))
        if kind == "VOTE":
            return SimpleNamespace(content="VOTE ЗА Поддерживаю зафиксированный финансовый план.")
        return SimpleNamespace(content=f"{kind}: позиция стороны {self.name} по очистным сооружениям.")


class LongReportPlayerModel(PlayerModel):
    def invoke(self, messages):
        if "ТВОЁ ДЕЙСТВИЕ: REPORT" in messages[-1].content:
            self.prompts.append(messages[-1].content)
            return SimpleNamespace(content="РАННИЙ_АРГУМЕНТ " + "длинный доклад " * 1200)
        return super().invoke(messages)


class JudgeModel:
    def __init__(self, close_rounds=()):
        self.prompts = []
        self.close_rounds = close_rounds

    def invoke(self, _messages):
        prompt = _messages[-1].content
        self.prompts.append(prompt)
        if "Подготовь судье записку перед действием:" in prompt:
            return SimpleNamespace(content=(
                "Факты: опирайся на данные события.\n"
                "Вопросы: проверь неподтверждённые заявления.\n"
                "Рамки решения: действуй в пределах доступных этапов."
            ))
        if '"next_stage"' in prompt:
            round_number = int(re.search(r"Круг: (\d+)", prompt).group(1))
            deficit = float(re.search(r"Дефицит, рубли: ([\d.]+)", prompt).group(1))
            contributions = json.loads(re.search(r"Засчитанные взносы, рубли: ({[^\n]+})", prompt).group(1))
            names = "; ".join(f"{name}: {(amount or 0) / 1_000_000:g} млн руб."
                              for name, amount in contributions.items())
            stage = ("close" if round_number in self.close_rounds else
                     "second_round" if round_number == 1 and deficit else "vote")
            return SimpleNamespace(content=json.dumps({
                "next_stage": stage,
                "remark": f"Оглашаю взносы: {names}; дефицит {deficit / 1_000_000:g} млн руб. "
                          f"Следующий этап: {stage}.",
            }, ensure_ascii=False))
        if "Событие:" in prompt:
            if "огласи засчитанные взносы" in prompt:
                facts = json.loads(prompt.split("Факты: ", 1)[1].split("\n", 1)[0])
                names = "; ".join(f"{name}: {(amount or 0) / 1_000_000:g} млн руб."
                                  for name, amount in facts["contributions_rub"].items())
                return SimpleNamespace(content=f"Оглашаю взносы: {names} Итого "
                                       f"{facts['total_rub'] / 1_000_000:g} млн руб.; "
                                       f"дефицит {facts['deficit_rub'] / 1_000_000:g} млн руб.")
            return SimpleNamespace(content="Продолжаем заседание по установленному порядку.")
        return SimpleNamespace(content='{"accepted": true, "remark": "Обоснование принято"}')


class MeetingProtocolTests(unittest.TestCase):
    def test_prompt_templates_load_and_render(self):
        self.assertIn("Ты участник", read_prompt("participant_system"))
        self.assertIn("Иван", render_prompt("question", speaker="Иван",
                                          focus="срок исполнения"))
        self.assertIn("открой круг", judge_event("open_round"))
        self.assertIn("judge_speech", JUDGE_PROMPTS)
        self.assertIn("participant_turn", PLAYER_PROMPTS)
        self.assertNotIn("role_holding", PLAYER_PROMPTS)
        self.assertEqual(set(HOLDING_PROMPTS), {"role_holding"})

    def test_exported_flow_prompts_match_prompt_config(self):
        flow = json.loads((ROOT / "LF_Agent_RAG.json").read_text(encoding="utf-8"))
        nodes = {node["id"]: node["data"]["node"]["template"]
                 for node in flow["data"]["nodes"]}
        self.assertEqual(nodes["PromptNode-3pqbR"]["template"]["value"],
                         read_prompt("meeting_rules"))
        for node_id, key in (
            ("PlayerAgent-sH7VB", "role_holding"),
            ("PlayerAgent-8iJzb", "role_admin"),
            ("PlayerAgent-mIWBs", "role_residents"),
            ("PlayerAgent-BJEKG", "role_region"),
            ("PlayerAgent-ZepWF", "role_ngo"),
        ):
            self.assertEqual(nodes[node_id]["prompt"]["value"], read_prompt(key))
        self.assertGreaterEqual(len(PROMPTS), 30)

    def test_duplicate_question_header_is_removed(self):
        clean = dialog_namespace["_clean_speech_message"]
        self.assertEqual(
            clean("Иван ➔ @Пётр (Вопрос): «Кто оплатит анализы?"),
            "Кто оплатит анализы?",
        )

    def make_game(self, pledges, malformed=None, semicolon_pledge=False, close_rounds=()):
        game = TwoLLMContractGame()
        game.use_rag = False
        game.rules_prompt = "Два круга и голосование."
        game.judge = {"service": MeetingJudgeService(JudgeModel(close_rounds))}
        game.players = []
        for index, amounts in enumerate(pledges):
            name = f"Игрок {index + 1}"
            limit = ([650, 250, 100, 10, 0][index] if index < 5 else 0) * 1_000_000
            game.players.append({
                "player_name": name,
                "prompt": "Защищай интересы своей стороны.",
                "llm": PlayerModel(name, amounts, 2 if malformed == index else None,
                                   semicolon_pledge=semicolon_pledge),
                "money": limit,
                "attrs": {"max_money": limit, "min_share": 0,
                          "comfort_share": 0, "comfort_money": 0},
            })
        return game

    def test_first_round_funds_project_and_skips_second(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        result = game.run_chat().data
        self.assertEqual(result["rounds_completed"], 1)
        self.assertEqual(result["total_money"], 1_000_000_000)
        self.assertTrue(result["voting_summary"]["passed"])
        self.assertIn("Игрок 1: 650 млн руб.", result["announcements"][0]["remark"])
        self.assertEqual(sum(h["action"]["type"] == "QUESTION" for h in result["history"]), 20)
        self.assertEqual(sum(h["action"]["type"] == "ANSWER" for h in result["history"]), 5)

    def test_second_round_ends_even_with_deficit_and_rejected_pledge(self):
        game = self.make_game([[400, 500], [150, 200], [30, 50], [0, 0], [0, 0]], malformed=0)
        result = game.run_chat().data
        self.assertEqual(result["rounds_completed"], 2)
        self.assertEqual(result["state"]["Игрок 1"]["contribution"], 400_000_000)
        self.assertEqual(result["deficit"], 350_000_000)
        self.assertFalse(result["voting_summary"]["passed"])
        self.assertEqual(sum(h["action"]["type"] == "QUESTION" for h in result["history"]), 40)
        self.assertTrue(any(h["action"]["type"] == "REJECTED" for h in result["history"]))
        self.assertFalse(game.judge["service"].decide(5, 5, 999_999_999, 1_000_000_000))

    def test_participant_count_is_not_fixed_at_five(self):
        for amounts in (
            [[650, 650], [250, 250], [100, 100]],
            [[650, 650], [250, 250], [100, 100], [0, 0], [0, 0], [0, 0]],
        ):
            with self.subTest(participants=len(amounts)):
                result = self.make_game(amounts).run_chat().data
                count = len(amounts)
                self.assertEqual(result["voting_summary"]["quorum"], count)
                self.assertTrue(result["voting_summary"]["passed"])
                self.assertEqual(sum(h["action"]["type"] == "QUESTION" for h in result["history"]),
                                 count * (count - 1))
        decide = MeetingJudgeService.decide
        self.assertFalse(decide(3, 6, 1_000_000_000, 1_000_000_000))
        self.assertTrue(decide(4, 6, 1_000_000_000, 1_000_000_000))

    def test_judge_can_close_after_first_round_without_starting_second(self):
        game = self.make_game([[400, 400], [150, 150], [30, 30]], close_rounds=(1,))
        result = game.run_chat().data
        self.assertEqual(result["rounds_completed"], 1)
        self.assertEqual(result["announcements"][0]["next_stage"], "close")
        self.assertFalse(result["voting_summary"]["held"])
        self.assertFalse(any(h["action"]["type"] == "VOTE" for h in result["history"]))

    def test_judge_can_close_after_second_round_with_deficit(self):
        game = self.make_game([[400, 400], [150, 150], [30, 30]], close_rounds=(2,))
        result = game.run_chat()
        self.assertEqual(result.data["rounds_completed"], 2)
        self.assertEqual([a["next_stage"] for a in result.data["announcements"]],
                         ["second_round", "close"])
        self.assertFalse(result.data["voting_summary"]["held"])
        self.assertIn("слушания завершены без голосования", result.text)

    def test_judge_cannot_close_in_speech_and_start_another_round(self):
        model = SimpleNamespace(invoke=lambda _messages: SimpleNamespace(content=json.dumps({
            "next_stage": "second_round", "remark": "Заседание окончено.",
        }, ensure_ascii=False)))
        judge = MeetingJudgeService(model)
        with self.assertRaisesRegex(ValueError, "Речь судьи завершает заседание"):
            judge.choose_stage(1, {"Игрок 1": {"contribution": 0}}, 1_000_000_000)

    def test_semicolon_pledges_are_counted_with_clean_explanations(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]],
                              semicolon_pledge=True)
        result = game.run_chat()
        self.assertEqual(result.data["total_money"], 1_000_000_000)
        self.assertEqual(result.data["rounds_completed"], 1)
        self.assertEqual(result.data["state"]["Игрок 2"]["contribution"], 250_000_000)
        self.assertFalse(any(h["action"]["type"] == "REJECTED" for h in result.data["history"]))
        self.assertNotIn("; SPEAK", result.text)
        self.assertNotIn("руб..", result.text)

    def test_questioners_receive_distinct_topics_and_prior_questions(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        game.run_chat()
        questions = [next(p for p in item["llm"].prompts
                          if "ТВОЁ ДЕЙСТВИЕ: QUESTION" in p)
                     for item in game.players[1:]]
        for topic, prompt in zip(("источник и предел финансирования", "срок и этап исполнения",
                                  "ответственный исполнитель", "независимая проверка"), questions):
            self.assertIn(topic, prompt)
        self.assertIn("[4] Игрок 2", questions[1])

    def test_early_history_is_available_to_late_participant_and_judge(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        first = game.players[0]
        first["llm"] = LongReportPlayerModel(first["player_name"], [650, 650])
        game.run_chat()
        late_vote = next(p for p in game.players[-1]["llm"].prompts
                         if "ТВОЁ ДЕЙСТВИЕ: VOTE" in p)
        self.assertIn("РАННИЙ_АРГУМЕНТ", late_vote)
        self.assertTrue(any("РАННИЙ_АРГУМЕНТ" in p and "Событие:" in p
                            for p in game.judge["service"].llm.prompts))

    def test_full_funding_claim_detector(self):
        claims = dialog_namespace["_claims_full_funding"]
        self.assertTrue(claims("Это позволит закрыть дефицит бюджета."))
        self.assertFalse(claims("Мы готовы выделить 150 млн рублей."))

    def test_judge_receives_discussion_and_prior_pledges(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        game.run_chat()
        prompts = [p for p in game.judge["service"].llm.prompts if "Предложение:" in p]
        self.assertIn("Игрок 1 (REPORT)", prompts[0])
        self.assertIn("Игрок 2 (QUESTION)", prompts[0])
        self.assertIn('"Игрок 1": 650000000.0', prompts[1])
        self.assertIn("Предложение:", prompts[1])

    def test_judge_speaks_from_events_and_exact_facts(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        result = game.run_chat().data
        prompts = game.judge["service"].llm.prompts
        self.assertTrue(any("Событие: открой круг" in p for p in prompts))
        self.assertTrue(any("Событие: предоставь слово" in p for p in prompts))
        self.assertTrue(any("Событие: огласи окончательное решение" in p for p in prompts))
        self.assertIn("Игрок 1: 650 млн руб.", result["announcements"][0]["remark"])

    def test_prompt_log_contains_every_role_and_judge_in_order(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100], [0, 0], [0, 0]])
        log = game.run_chat().data["prompt_logs"]
        self.assertEqual(log[0]["role"], "Секретарь судьи")
        self.assertEqual(log[1]["role"], "Судья")
        self.assertEqual(log[2]["role"], "Секретарь судьи")
        self.assertEqual(log[3]["role"], "Судья")
        self.assertEqual(log[4]["role"], "Секретарь для Игрок 1")
        self.assertEqual(log[5]["role"], "Игрок 1")
        self.assertEqual({entry["role"] for entry in log},
                         {"Судья", "Секретарь судьи",
                          *(f"Игрок {index}" for index in range(1, 6)),
                          *(f"Секретарь для Игрок {index}" for index in range(1, 6))})
        self.assertTrue(all(entry["system_prompt"] and entry["user_prompt"] for entry in log))
        self.assertTrue(any(entry["role"] == "Судья" and
                            entry["action"] == "проверка взноса" for entry in log))

    def test_judge_secretary_precedes_speech_review_and_transition(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100]])
        log = game.run_chat().data["prompt_logs"]
        judge_actions = [item for item in log if item["role"] == "Судья"]
        self.assertTrue(any(item["action"] == "проверка взноса" for item in judge_actions))
        self.assertTrue(any(item["action"] == "выбор этапа" for item in judge_actions))
        for index, item in enumerate(log):
            if item["role"] == "Судья":
                self.assertEqual(log[index - 1]["role"], "Секретарь судьи")
                self.assertIn("Записка секретаря судьи:", item["user_prompt"])

    def test_secretary_briefs_each_role_before_every_action(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100]])
        result = game.run_chat().data
        for player in game.players:
            prompts = player["llm"].prompts
            briefs = [p for p in prompts if "Подготовь для " in p and "краткую записку" in p]
            actions = [p for p in prompts if "ТВОЁ ДЕЙСТВИЕ:" in p]
            self.assertEqual(len(briefs), len(actions))
            self.assertTrue(all("ЗАПИСКА ЛИЧНОГО СЕКРЕТАРЯ:" in p for p in actions))
            self.assertIn("Последствия: соблюдай свой бюджет", actions[0])
        self.assertEqual(sum(p["action"] == "BRIEF" and p["role"].startswith("Секретарь для")
                             for p in result["prompt_logs"]),
                         sum(p["action"] in {"REPORT", "QUESTION", "ANSWER", "PLEDGE", "VOTE"}
                             for p in result["prompt_logs"]))

    def test_player_sees_other_players_speeches_not_judge_or_ledger(self):
        game = self.make_game([[650, 650], [250, 250], [100, 100]])
        game.run_chat()
        player = game.players[1]["llm"]
        report = next(p for p in player.prompts if "ТВОЁ ДЕЙСТВИЕ: REPORT" in p)
        secretary = next(p for p in player.prompts if "Подготовь для " in p)
        self.assertIn("Игрок 1: REPORT: позиция стороны Игрок 1", report)
        self.assertNotIn("Судья:", report)
        self.assertNotIn("СОСТОЯНИЕ ВЗНОСОВ:", report)
        self.assertNotIn("Публичная история заседания:", report)
        self.assertIn("Публичное состояние взносов:", secretary)
        self.assertIn("Не уступай без причины", secretary)
        first_secretary = game.players[0]["llm"].prompts[0]
        self.assertIn("включая текущий: 15", first_secretary)
        self.assertIn("включая текущий: 3", next(
            p for p in game.players[0]["llm"].prompts
            if "перед действием VOTE" in p
        ))


if __name__ == "__main__":
    unittest.main()
