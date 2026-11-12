"""Show sample LLM messages for every dialog phase without calling an LLM.

Run: python inspect_dialog_prompts.py
"""

import ast
import copy
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SOURCE = Path(__file__).parent / "langflow/custom_components/my_components/dialog.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
names = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
game = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NegotiationGame")
methods = {node.name: node for node in game.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
helpers = ["is_owner", "format_money", "render_profile", "render_player_levers",
           "render_calculator", "render_state", "render_draft_treaty",
           "render_voter_checklist", "render_history", "player_turn_examples"]
scope = {"copy": copy, "json": json}
scope["DEFAULT_PROJECT_COST"] = 270_000_000.0
exec(compile(ast.Module(body=[names[name] for name in helpers], type_ignores=[]), str(SOURCE), "exec"), scope)


def expression(method_name, variable, context):
    method = methods[method_name]
    assignment = next(node for node in ast.walk(method)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets))
    return eval(compile(ast.Expression(assignment.value), str(SOURCE), "eval"), scope | context)


settings = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_get_default_settings")
exec(compile(ast.Module(body=[settings], type_ignores=[]), str(SOURCE), "exec"), scope)
levers = scope["_get_default_settings"]()["levers"]
speaker = "Администрация города N"
other = "Промышленный холдинг"
profile = {"money": 100_000_000, "attrs": {"min_share": 20, "comfort_share": 30,
           "comfort_money": 70_000_000, "red_lines": ["Питьевая вода жителям"]}}
state = {"player_ids": {"city": speaker, "holding": other},
         "players": {speaker: {"contribution": 50_000_000, "share": 25, "spent_on_levers": 0},
                     other: {"contribution": 100_000_000, "share": 75, "spent_on_levers": 0}},
         "levers": {lever["id"]: lever.get("default") for lever in levers},
         "demands": [{"from": other, "target": speaker, "status": "pending",
                      "demand_type": "pledge", "title": "Увеличить взнос", "value": 70_000_000}]}
history = [{"speaker": other, "turn": 1, "action": {"type": "ACTION", "message": "Предлагаю увеличить взнос", "applied": []}}]
total_cost = 270_000_000
deficit = total_cost - sum(player["contribution"] for player in state["players"].values())
common = dict(speaker=speaker, player_prompt="Тестовая роль", profile=profile, state=state,
              history=history, rules_prompt="Тестовый регламент", levers_config=levers,
              total_cost=total_cost)

phases = [
    ("Первая фаза", "_build_prompt_phase1", "_run_phase1", {"turn": 1, "total_phase1": 2}, {}),
    ("Переговоры", "_build_prompt_phase2", "_run_phase2",
     {"turn_in_phase": 1, "max_phase_turns": 15, "global_turn": 3},
     {"deficit_clause": f"⚠️ Дефицит бюджета: {scope['format_money'](deficit)}. Торгуйся, выдвигай встречные требования или предлагай компромиссное финансирование."}),
    ("Последний шанс", "_build_prompt_last_call", "_run_last_call",
     {"turn_counter": 10, "deficit": deficit, "available_money": 50_000_000},
     {"cur_deficit": deficit}),
    ("Голосование", "_build_prompt_phase3", "_run_phase3",
     {"turn_in_phase": 1, "total_voters": 2, "majority_threshold": 2},
     {"deficit_info": f"Текущий статус бюджета проекта: ⚠️ ДЕФИЦИТ {scope['format_money'](deficit)}. ",
      "majority_threshold": 2, "num_players": 2}),
]

for title, builder, runner, kwargs, extra in phases:
    method = methods[builder]
    func = ast.FunctionDef(name=builder, args=method.args, body=method.body,
                           decorator_list=[], returns=method.returns, type_comment=None)
    ast.fix_missing_locations(func)
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(SOURCE), "exec"), scope)
    prompt = scope[builder](**{key: value for key, value in (common | kwargs).items()
                               if key in {arg.arg for arg in method.args.args}})
    system = expression(runner, "system_message", extra)
    print(f"\n{'=' * 24} {title} {'=' * 24}")
    voting = runner == "_run_phase3"
    contract_levers = [{**lever, "owner": next(
        (name for name in state["player_ids"].values() if scope["is_owner"](lever.get("owner", ""), name)),
        lever.get("owner"))} for lever in levers]
    contract = expression("_structured_turn", "contract", {
        "state": state, "contract_levers": contract_levers, "voting": voting,
        "current": {"name": speaker}})
    rag = ""  # _rag_context returns an empty string when RAG is not configured.
    request = [("system", common["player_prompt"]),
               ("human", "РЕГЛАМЕНТ ЗАСЕДАНИЯ:\n" + common["rules_prompt"]),
               ("human", contract), ("human", system), ("human", prompt + "\n\n" + rag)]
    for index, (role, content) in enumerate(request, 1):
        print(f"\n[{index}] {role.upper()}\n{content}")
    print("\n[structured output] with_structured_output(PlayerTurn, method=function_calling); "
          "схема инструмента передаётся модели отдельно от messages.")
