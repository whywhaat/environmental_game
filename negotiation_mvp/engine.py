"""Round orchestration; STATE mutation occurs only after every speech validates."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal
from typing import Any, Mapping

from config import prompts

from .core import Assignment, GameConfig, GameError, State, funding_total, is_complete, merge_round, parse_sets, render_table, score_state
from .llm import ChatModel


class ConversionError(RuntimeError):
    pass


COMMON_PLAYER_SYSTEM_PROMPT = prompts.PLAYER_SYSTEM_PROMPT


def _verify_numeric_evidence(speech: str, actions: list[Assignment]) -> None:
    """Reject numeric values with no digit evidence in the source speech."""
    numbers = {int(re.sub(r"\D", "", token)) for token in re.findall(r"\d[\d\s.,]*", speech) if re.sub(r"\D", "", token)}
    for action in actions:
        if action.path.endswith(".contribution"):
            amount = int(action.value)
            if amount not in numbers and not (amount % 1_000_000 == 0 and amount // 1_000_000 in numbers):
                raise GameError(f"Contribution {amount} lacks numeric evidence in speech")


def _drop_unsupported_numeric_sets(speech: str, raw: str, actions: list[Assignment]) -> tuple[str, list[Assignment]]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    kept = []
    for line, action in zip(lines, actions):
        try:
            _verify_numeric_evidence(speech, [action])
        except GameError:
            continue
        kept.append((line, action))
    return "\n".join(line for line, _ in kept) or "NO_CHANGES", [action for _, action in kept]


def _closed_schema_sets(raw: str, config: GameConfig, speaker: str, voting: bool) -> str:
    allowed = {f"participants.{speaker}.vote"} if voting else set(config.required_paths)
    kept = []
    for line in raw.splitlines():
        match = re.fullmatch(r"SET\s+([a-z][a-z0-9_.]*)\s*=\s*.+", line.strip())
        if not match or match.group(1) in allowed:
            kept.append(line)
    return "\n".join(kept).strip() or "NO_CHANGES"


def _drop_ambiguous_contributions(raw: str) -> str:
    """Keep unambiguous SET lines when a model repeats a contribution field."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    values: dict[str, set[str]] = {}
    for line in lines:
        match = re.fullmatch(r"SET\s+(participants\.[a-z][a-z0-9_]*\.contribution)\s*=\s*(.+)", line)
        if match:
            values.setdefault(match.group(1), set()).add(match.group(2))
    ambiguous = {path for path, entries in values.items() if len(entries) > 1}
    seen: set[str] = set()
    kept = []
    for line in lines:
        match = re.fullmatch(r"SET\s+(participants\.[a-z][a-z0-9_]*\.contribution)\s*=\s*(.+)", line)
        if match:
            path = match.group(1)
            if path in ambiguous or path in seen:
                continue
            seen.add(path)
        kept.append(line)
    return "\n".join(kept) or "NO_CHANGES"


def _allowed_paths(config: GameConfig, speaker: str, voting: bool) -> str:
    if voting:
        return f"participants.{speaker}.vote = true|false"
    lines = [f"participants.{p.id}.contribution = <integer rubles>" for p in config.players]
    lines += [f"components.{c.id}.responsible = <player id>" for c in config.components]
    return "\n".join(lines)


_MONEY_RE = re.compile(
    r"([+-]?(?:\d{1,3}(?:[\s,._]\d{3})+|\d+(?:[.,]\d+)?))\s*(млн|тыс)?\s*(?:₽|руб(?:лей|ля|ль|\.)?)?",
    re.IGNORECASE,
)


def _player_aliases(config: GameConfig) -> dict[str, str]:
    aliases = {player.id.casefold(): player.id for player in config.players}
    aliases.update({player.name.casefold(): player.id for player in config.players})
    prefixes: dict[str, list[str]] = {}
    for player in config.players:
        prefixes.setdefault(player.id.split("_", 1)[0].casefold(), []).append(player.id)
    aliases.update({prefix: ids[0] for prefix, ids in prefixes.items() if len(ids) == 1 and prefix not in aliases})
    return aliases


def _normalize_set_value(path: str, value: object, aliases: dict[str, str]) -> object:
    if not isinstance(value, str):
        if path.endswith(".contribution") and isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return value
    cleaned = " ".join(value.strip().strip("'\"").split())
    if cleaned.casefold() in {"no_changes", "none", "null"} or (path.endswith(".responsible") and cleaned.casefold() == "мимо"):
        return None
    if path.endswith(".responsible"):
        return aliases.get(cleaned.casefold(), cleaned)
    if path.endswith(".vote"):
        return {"true": True, "false": False, "за": True, "против": False}.get(cleaned.casefold(), cleaned)
    if path.endswith(".contribution"):
        match = _MONEY_RE.fullmatch(cleaned)
        if match:
            number = match.group(1)
            if re.fullmatch(r"[+-]?\d{1,3}(?:[\s,._]\d{3})+", number):
                number = re.sub(r"[\s,._]", "", number)
            else:
                number = number.replace(",", ".")
            amount = Decimal(number) * {None: 1, "млн": 1_000_000, "тыс": 1_000}[match.group(2).casefold() if match.group(2) else None]
            if amount == amount.to_integral_value():
                return int(amount)
    return cleaned


def _normalize_converter_sets(raw: str, config: GameConfig) -> str:
    """Normalize values by field type without accepting unknown or ambiguous data."""
    aliases = _player_aliases(config)
    normalized = []
    for line in raw.splitlines():
        match = re.fullmatch(r"(SET\s+([a-z][a-z0-9_.]*)\s*=\s*)(.+)", line.strip())
        if not match:
            normalized.append(line)
            continue
        prefix, path, value_text = match.groups()
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError:
            value = value_text
        value = _normalize_set_value(path, value, aliases)
        if value is None:
            continue
        normalized.append(prefix + json.dumps(value, ensure_ascii=False))
    lines = [line.strip() for line in normalized if line.strip() and line.strip() != "NO_CHANGES"]
    return "\n".join(lines) or "NO_CHANGES"


def convert_speech(chat: ChatModel, model: str, speech: str, speaker: str, config: GameConfig, seq: int, voting: bool = False) -> tuple[str, list[Assignment]]:
    prompt = (prompts.CONVERTER_VOTE_PROMPT.format(speaker=speaker) if voting else "") + prompts.CONVERTER_USER_PROMPT.format(
        allowed_paths=_allowed_paths(config, speaker, voting),
        example_component_id=config.components[0].id,
        example_player_id=config.players[0].id,
        players=", ".join(f"{p.id}={p.name}" for p in config.players),
        speaker=speaker,
        speech=speech,
    )
    last_error = ""
    last_output = ""
    for attempt in range(2):
        user_prompt = prompt + (prompts.CONVERTER_RETRY_PROMPT.format(error=last_error, previous_output=last_output[:2000]) if attempt else "")
        messages = [
            {"role": "system", "content": prompts.CONVERTER_SYSTEM_PROMPT + "\n\n" + prompts.CONVERTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + "\n\n" + user_prompt},
        ]
        raw = _normalize_converter_sets(chat.complete(model, messages), config)
        try:
            actions = parse_sets(raw, speaker, config, seq)
            if voting:
                if len(actions) > 1 or (actions and actions[0].path != f"participants.{speaker}.vote"):
                    raise GameError("Voting may set only the current player's vote")
            elif any(action.path.endswith(".vote") for action in actions):
                raise GameError("Votes are not allowed during negotiation")
            else:
                _verify_numeric_evidence(speech, actions)
            return raw, actions
        except GameError as exc:
            last_output = raw
            if attempt == 1 and not voting and "lacks numeric evidence in speech" in str(exc) and "lacks numeric evidence in speech" in last_error:
                return _drop_unsupported_numeric_sets(speech, raw, actions)
            if attempt == 1 and not voting and last_error.startswith("Duplicate SET field in one speech: participants.") and str(exc).startswith("Duplicate SET field in one speech: participants."):
                cleaned = _drop_ambiguous_contributions(raw)
                try:
                    actions = parse_sets(cleaned, speaker, config, seq)
                    _verify_numeric_evidence(speech, actions)
                    return cleaned, actions
                except GameError as cleaned_error:
                    last_error = str(cleaned_error)
                    continue
            if attempt == 1 and not voting and last_error.startswith("Unknown SET path:") and str(exc).startswith("Unknown SET path:"):
                cleaned = _closed_schema_sets(raw, config, speaker, voting)
                try:
                    actions = parse_sets(cleaned, speaker, config, seq)
                    _verify_numeric_evidence(speech, actions)
                    return cleaned, actions
                except GameError as cleaned_error:
                    last_error = str(cleaned_error)
                    continue
            last_error = str(exc)
    raise ConversionError(f"Converter failed twice for {speaker}: {last_error}; last output: {raw[:500]!r}")


def _player_messages(player, config: GameConfig, session: list[dict[str, str]], rag_context: str = "") -> list[dict[str, str]]:
    names = {participant.id: participant.name for participant in config.players}
    expectations = prompts.player_expectations(
        player.expected_own_share,
        config.total_cost,
        [(component.name, names[player.desired_responsible[component.id]]) for component in config.components],
    )
    private_history = [{"role": "user", "content": prompts.PERSONAL_HISTORY_PROMPT.format(event=event)} for event in player.personal_history]
    if player.motivation:
        private_history.append({"role": "user", "content": prompts.MOTIVATION_PROMPT.format(motivation=player.motivation)})
    private_history.append({"role": "user", "content": expectations})
    context = player.personal_context
    if getattr(player, "prompt", None) and player.prompt != player.personal_context:
        context = f"{player.personal_context}\n\n### Стратегия и инструкция:\n{player.prompt}" if player.personal_context else player.prompt
    player_prompts = [{"role": "system", "content": COMMON_PLAYER_SYSTEM_PROMPT + "\n\n" + context}] + private_history
    if rag_context:
        player_prompts.append({"role": "user", "content": rag_context})
    history = [
        {"role": "assistant" if event.get("player") == player.id else "user", "content": event["text"]} for event in session
    ]
    return player_prompts + history


def _round_notice(number: int, limit: int) -> str:
    return prompts.round_notice(number, limit)


def run_game(config: GameConfig, chat: ChatModel, player_model: str | Mapping[str, str], converter_model: str, rag_retriever: Any | None = None) -> dict[str, Any]:
    def model_for(player_id: str) -> str:
        return player_model[player_id] if isinstance(player_model, Mapping) else player_model
    state = State({path: [] for path in config.required_paths})
    turns: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    session: list[dict[str, str]] = [
        {"kind": "opening", "text": prompts.opening_message([player.name for player in config.players], [(component.name, component.cost) for component in config.components], config.total_cost)},
        {"kind": "rules", "text": prompts.GAME_RULES_PROMPT},
        {"kind": "table", "text": render_table(state, config)},
    ]
    seq = 1
    for round_number in range(1, config.max_rounds + 1):
        proposals: list[Assignment] = []
        speeches: list[dict[str, str]] = []
        session.append({"kind": "clock", "text": _round_notice(round_number, config.max_rounds)})
        for player in config.players:
            session.append({"kind": "floor", "text": prompts.ROUND_FLOOR_PROMPT.format(round_number=round_number, player_name=player.name)})
            recent = session[-2]["text"] if len(session) >= 2 else ""
            rag_context = rag_retriever.get_context(player.id, round_number, recent) if rag_retriever else ""
            speech = chat.complete(model_for(player.id), _player_messages(player, config, session, rag_context=rag_context))
            raw_sets, actions = convert_speech(chat, converter_model, speech, player.id, config, seq)
            seq += len(actions)
            proposals.extend(actions)
            speeches.append({"speaker": player.id, "speech": speech, "sets": raw_sets})
            session.append({"kind": "speech", "player": player.id, "text": prompts.SPEECH_PROMPT.format(player_name=player.name, speech=speech)})
        # Atomic round commit: all conversion and validation finished above.
        state = merge_round(state, proposals, config)
        scores = score_state(state, config)
        snapshot = {"number": round_number, "speeches": speeches, "state": state.to_dict(), "table": render_table(state, config), "scores": scores, "funding_total": funding_total(state, config), "conflicts": state.conflicts()}
        rounds.append(snapshot)
        turns.extend(speeches)
        session.append({"kind": "table", "text": snapshot["table"]})
        if is_complete(state, config):
            break
    session.append({"kind": "stage", "text": prompts.VOTING_OPENED_PROMPT})
    votes: dict[str, bool | None] = {}
    vote_actions: list[Assignment] = []
    for player in config.players:
        session.append({"kind": "floor", "text": prompts.VOTE_FLOOR_PROMPT.format(player_name=player.name)})
        recent = session[-2]["text"] if len(session) >= 2 else ""
        rag_context = rag_retriever.get_context(player.id, config.max_rounds + 1, "Финальное голосование по договору: " + recent) if rag_retriever else ""
        speech = chat.complete(model_for(player.id), _player_messages(player, config, session, rag_context=rag_context))
        raw_sets, actions = convert_speech(chat, converter_model, speech, player.id, config, seq, voting=True)
        seq += len(actions)
        vote = actions[0].value if actions else None
        votes[player.id] = vote
        vote_actions.extend(actions)
        turns.append({"speaker": player.id, "speech": speech, "sets": raw_sets, "phase": "vote"})
        session.append({"kind": "vote", "player": player.id, "text": prompts.SPEECH_PROMPT.format(player_name=player.name, speech=speech)})
    for action in vote_actions:
        state.fields[action.path] = [{"value": action.value, "player": action.player, "seq": action.seq}]
    accepted = sum(vote is True for vote in votes.values()) > len(config.players) / 2 and is_complete(state, config)
    return {"total_cost": config.total_cost, "rounds": rounds, "final_state": state.to_dict(), "final_table": render_table(state, config, signed=accepted), "final_scores": score_state(state, config, signed=accepted), "votes": votes, "accepted": accepted, "turns": turns, "session": session, "rag_audit": getattr(rag_retriever, "audit", []) if rag_retriever else []}
