# -*- coding: utf-8 -*-
import sys
import copy
import os

from types import ModuleType

# Mock lfx packages
m_lfx = ModuleType("lfx")
m_lfx.__path__ = []
sys.modules["lfx"] = m_lfx

m_custom = ModuleType("lfx.custom")
class Component: pass
m_custom.Component = Component
sys.modules["lfx.custom"] = m_custom

m_io = ModuleType("lfx.io")
m_io.DataInput = lambda **kw: kw
m_io.HandleInput = lambda **kw: kw
m_io.IntInput = lambda **kw: kw
m_io.MultilineInput = lambda **kw: kw
m_io.Output = lambda **kw: kw
sys.modules["lfx.io"] = m_io

m_schema = ModuleType("lfx.schema")
m_schema.__path__ = []
sys.modules["lfx.schema"] = m_schema

m_data = ModuleType("lfx.schema.data")
m_data.Data = dict
sys.modules["lfx.schema.data"] = m_data

m_msg = ModuleType("lfx.schema.message")
m_msg.Message = dict
sys.modules["lfx.schema.message"] = m_msg

m_lc = ModuleType("langchain_core.messages")
m_lc.HumanMessage = lambda content: {"role": "human", "content": content}
m_lc.SystemMessage = lambda content: {"role": "system", "content": content}
m_lc.AIMessage = lambda content: {"role": "ai", "content": content}
sys.modules["langchain_core.messages"] = m_lc

import importlib.util
spec = importlib.util.spec_from_file_location("dialog", "langflow/custom_components/my_components/dialog.py")
dialog = importlib.util.module_from_spec(spec)
sys.modules["dialog"] = dialog
spec.loader.exec_module(dialog)

_clean_speech_message = dialog._clean_speech_message
TwoLLMContractGame = dialog.TwoLLMContractGame
_format_money = dialog._format_money

print("=== 1. Testing _clean_speech_message ===")
msg_with_brackets = "[SET me Взнос 0]\n[SET @Промкомплекс 70%]\nПриветствую всех участников слушаний!"
cleaned = _clean_speech_message(msg_with_brackets)
print("Cleaned result:", repr(cleaned))
assert "[SET" not in cleaned, "Brackets not cleaned!"
assert "Приветствую всех участников слушаний!" in cleaned

print("\n=== 2. Testing _parse_single_set self-demand redirection ===")
levers_config = [
    {"id": "moratorium", "title": "Мораторий на проверки бизнеса", "owner": "Регион", "type": "bool", "aliases": ["мораторий"]},
    {"id": "ndt", "title": "Внедрение НДТ", "owner": "Промкомплекс", "type": "bool", "aliases": ["ндт"]},
]
all_names = ["Администрация", "Эко-НКО", "Жители", "Регион", "Промкомплекс"]

# Жители writing SET @Жители мораторий да (they don't own it)
cmd = "SET @Жители мораторий да"
dummy_state = {"players": {n: {} for n in all_names}}
parsed = TwoLLMContractGame._parse_single_set(cmd, "Жители", dummy_state, levers_config, 270_000_000.0)
print("Parsed single set for non-owner targeting self:", parsed)
assert parsed["target"] == "Регион", f"Expected target to be 'Регион', got '{parsed['target']}'"

# Промкомплекс writing SET @Промкомплекс ндт да (they DO own it)
cmd2 = "SET @Промкомплекс ндт да"
parsed2 = TwoLLMContractGame._parse_single_set(cmd2, "Промкомплекс", dummy_state, levers_config, 270_000_000.0)
print("Parsed single set for owner targeting self:", parsed2)
assert parsed2["type"] == "LEVER_SET", f"Expected LEVER_SET, got '{parsed2['type']}'"

print("\n=== 3. Testing _apply_actions self-demand interception ===")
state = {
    "players": {
        "Жители": {"contribution": 0.0, "share": 0.0, "spent_on_levers": 0.0},
        "Регион": {"contribution": 50_000_000.0, "share": 20.0, "spent_on_levers": 0.0},
        "Промкомплекс": {"contribution": 150_000_000.0, "share": 60.0, "spent_on_levers": 0.0},
        "Администрация": {"contribution": 30_000_000.0, "share": 10.0, "spent_on_levers": 0.0},
        "Эко-НКО": {"contribution": 0.0, "share": 0.0, "spent_on_levers": 0.0},
    },
    "levers": {},
    "demands": []
}

# Demand targeting self on lever owned by self
action_self_owned = {
    "speaker": "Промкомплекс",
    "actions": [{
        "type": "DEMAND_SET",
        "demand_type": "lever",
        "from": "Промкомплекс",
        "target": "Промкомплекс",
        "lever_id": "ndt",
        "title": "Внедрение НДТ",
        "value": True,
    }]
}
applied = TwoLLMContractGame._apply_actions(state, action_self_owned, target=None, total_cost=270_000_000.0, levers_config=levers_config)
print("Applied self-demand on owned lever:", applied)
assert applied["applied"][0]["kind"] == "LEVER", "Expected conversion to LEVER!"
assert state["levers"]["ndt"] is True

# Demand targeting self on pledge (should be dropped)
action_self_pledge = {
    "speaker": "Жители",
    "actions": [{
        "type": "DEMAND_PLEDGE",
        "demand_type": "pledge",
        "from": "Жители",
        "target": "Жители",
        "value": 10_000_000.0,
    }]
}
applied_p = TwoLLMContractGame._apply_actions(state, action_self_pledge, target=None, total_cost=270_000_000.0, levers_config=levers_config)
assert len(applied_p["applied"]) == 0, "Self pledge demand should be dropped!"

print("\n=== 4. Testing _render_dialogue with majority vote & 10m deficit ===")
voting_summary = {
    "records": [
        {"player": "Администрация", "vote": "ЗА", "comment": "Одобряем"},
        {"player": "Регион", "vote": "ЗА", "comment": "Одобряем"},
        {"player": "Промкомплекс", "vote": "ЗА", "comment": "Одобряем"},
        {"player": "Жители", "vote": "ЗА", "comment": "Одобряем"},
        {"player": "Эко-НКО", "vote": "ПРОТИВ", "comment": "Требуем больше гарантий"},
    ],
    "votes_for": 4,
    "votes_against": 1,
    "votes_abstain": 0,
    "quorum": 5,
    "majority_threshold": 3,
    "deficit": 10_000_000.0,
    "budget_closed": False,
    "passed": True,
}
rendered = TwoLLMContractGame._render_dialogue([], state, voting_summary, 270_000_000.0, levers_config)
print("\n--- Rendered Protocol Excerpt ---")
for line in rendered.splitlines()[-25:]:
    print(line)

assert "ПРОТОКОЛ ПРИНЯТ БОЛЬШИНСТВОМ ГОЛОСОВ" in rendered
assert "Примечание по финансированию" in rendered
assert "| **Жители** | 0 руб. | 0% |" in rendered

print("\n=== 5. Testing Phase 2 Speaker Rotation (Simulation) ===")
players = [{"name": n, "prompt": "p", "profile": {"money": 0}} for n in all_names]
name_to_idx = {p["name"]: idx for idx, p in enumerate(players)}
names = [p["name"] for p in players]

pending_speaker_idx = None
dialogue_pairs = []
turns_silent = {p["name"]: 0 for p in players}
last_speaker_name = None
max_turns = 15

spoken_count = {n: 0 for n in names}

for b_turn in range(max_turns):
    max_silent_turns = max(turns_silent.values()) if turns_silent else 0
    starved_player = None
    if max_silent_turns >= 3:
        candidates = [p_name for p_name in names if turns_silent[p_name] == max_silent_turns and p_name != last_speaker_name]
        if candidates:
            starved_player = candidates[0]

    if pending_speaker_idx is not None and len(dialogue_pairs) >= 2:
        last_s, last_t = dialogue_pairs[-1]
        if last_t is not None:
            target_pair = {last_s, last_t}
            recent_count = 0
            for s_name, t_name in reversed(dialogue_pairs):
                if t_name is not None and {s_name, t_name} == target_pair:
                    recent_count += 1
                else:
                    break
            if recent_count >= 2:
                pending_speaker_idx = None

    if starved_player is not None:
        pending_name = players[pending_speaker_idx]["name"] if pending_speaker_idx is not None else None
        if pending_name is None or turns_silent[starved_player] > turns_silent.get(pending_name, 0):
            current_idx = name_to_idx[starved_player]
            pending_speaker_idx = None
        else:
            current_idx = pending_speaker_idx
            pending_speaker_idx = None
    elif pending_speaker_idx is not None:
        if players[pending_speaker_idx]["name"] == last_speaker_name:
            eligible = [p_name for p_name in names if p_name != last_speaker_name]
            silent_longest = max(eligible, key=lambda p_name: (turns_silent[p_name], -name_to_idx[p_name]))
            current_idx = name_to_idx[silent_longest]
        else:
            current_idx = pending_speaker_idx
        pending_speaker_idx = None
    else:
        eligible = [p_name for p_name in names if p_name != last_speaker_name]
        if not eligible:
            eligible = names
        silent_longest = max(eligible, key=lambda p_name: (turns_silent[p_name], -name_to_idx[p_name]))
        current_idx = name_to_idx[silent_longest]

    current = players[current_idx]
    speaker = current["name"]
    last_speaker_name = speaker
    spoken_count[speaker] += 1

    for p_name in names:
        turns_silent[p_name] = 0 if p_name == speaker else turns_silent[p_name] + 1

    # Simulate speaker addressing a target:
    # Say Admin talks to Region, Region talks to Residents, Residents talks to Admin...
    if speaker == "Администрация":
        target = "Регион"
    elif speaker == "Регион":
        target = "Жители"
    elif speaker == "Жители":
        target = "Администрация"
    else:
        target = "Администрация"

    dialogue_pairs.append((speaker, target))
    pending_speaker_idx = name_to_idx[target]
    print(f"Turn {b_turn+1:2d}: {speaker:<15} -> @{target:<15} (Silent: {turns_silent})")

print("\nTotal speeches per player across 15 turns:", spoken_count)
for n in names:
    assert spoken_count[n] >= 2, f"Player {n} was starved! Only spoke {spoken_count[n]} times!"

print("\n🎉 ALL UNIT TESTS PASSED WITH 100% SUCCESS!")
