"""Round orchestration; STATE mutation occurs only after every speech validates."""

from __future__ import annotations

import re
from typing import Any

from .core import Assignment, GameConfig, GameError, State, funding_total, is_complete, merge_round, parse_sets, render_table, score_state
from .llm import ChatModel


class ConversionError(RuntimeError):
    pass


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


def _allowed_paths(config: GameConfig, speaker: str, voting: bool) -> str:
    if voting:
        return f"participants.{speaker}.vote = true|false"
    lines = [f"participants.{p.id}.contribution = <integer rubles>" for p in config.players]
    lines += [f"components.{c.id}.responsible = <player id>" for c in config.components]
    return "\n".join(lines)


def convert_speech(chat: ChatModel, model: str, speech: str, speaker: str, config: GameConfig, seq: int, voting: bool = False) -> tuple[str, list[Assignment]]:
    prompt = (
        "Извлеки только явно произнесённые решения из реплики в формате SET <path> = <JSON value>, по одной команде на строку. "
        "Если решений нет, ответь ровно NO_CHANGES; не выводи пояснений или Markdown, не придумывай значения. "
        "Денежные суммы указывай целыми рублями, имена игроков заменяй их id. "
        'Значение responsible ОБЯЗАТЕЛЬНО заключай в двойные кавычки, например SET components.water_supply.responsible = "industrial"; '
        "числа и true/false пиши без кавычек. "
        "Допустимые поля:\n" + _allowed_paths(config, speaker, voting) + "\n"
        "Игроки: " + ", ".join(f"{p.id}={p.name}" for p in config.players) + "\n"
        f"Автор реплики: {speaker}\nРеплика:\n{speech}"
    )
    last_error = ""
    for attempt in range(2):
        messages = [{"role": "system", "content": "Ты конвертер реплик в команды SET; отвечай строго указанным форматом."}, {"role": "user", "content": prompt + (f"\nПредыдущий ответ ошибочен: {last_error}. Исправь его." if attempt else "")}]
        raw = chat.complete(model, messages)
        try:
            actions = parse_sets(raw, speaker, config, seq)
            if voting:
                if len(actions) != 1 or actions[0].path != f"participants.{speaker}.vote":
                    raise GameError("Vote must set exactly the current player's vote")
            elif any(action.path.endswith(".vote") for action in actions):
                raise GameError("Votes are not allowed during negotiation")
            else:
                _verify_numeric_evidence(speech, actions)
            return raw, actions
        except GameError as exc:
            if attempt == 1 and not voting and "lacks numeric evidence in speech" in str(exc) and "lacks numeric evidence in speech" in last_error:
                return _drop_unsupported_numeric_sets(speech, raw, actions)
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


def _player_messages(player, config: GameConfig, session: list[dict[str, str]]) -> list[dict[str, str]]:
    names = {participant.id: participant.name for participant in config.players}
    obligations = [(component.id, component.name) for component in config.components]
    expected_rubles = f"{player.expected_own_share * config.total_cost:.2f}".rstrip("0").rstrip(".")
    expectations = (
        f"Личные ожидания для оценки: желаемый собственный взнос — {player.expected_own_share * 100:g}% "
        f"стоимости проекта ({expected_rubles} ₽). "
        "Желаемые ответственные: "
        + "; ".join(f"{label} — {names[player.desired_responsible[component_id]]}" for component_id, label in obligations)
        + "."
    )
    private_history = [{"role": "user", "content": f"Личная история: {event}"} for event in player.personal_history]
    private_history.append({"role": "user", "content": f"Мотивация: {player.motivation}"})
    private_history.append({"role": "user", "content": expectations})
    return [{"role": "system", "content": player.personal_context}] + private_history + [
        {"role": "assistant" if event.get("player") == player.id else "user", "content": event["text"]} for event in session
    ]


def _get_vote(chat: ChatModel, model: str, player, config: GameConfig, session: list[dict[str, str]]) -> tuple[str, bool]:
    speech = chat.complete(model, _player_messages(player, config, session))
    if re.search(r"\b(?:VOTE\s+|голосую\s+|голосуем\s+|мой\s+голос\s+)ПРОТИВ\b|\bне\s+поддержива", speech, re.IGNORECASE):
        return speech, False
    if re.search(r"\b(?:VOTE\s+|голосую\s+|голосуем\s+|мой\s+голос\s+)ЗА\b|\bподдержива", speech, re.IGNORECASE):
        return speech, True
    return speech, False


def _round_notice(number: int, limit: int) -> str:
    remaining = limit - number + 1
    unit = "раунд" if remaining % 10 == 1 and remaining % 100 != 11 else "раунда" if remaining % 10 in (2, 3, 4) and remaining % 100 not in (12, 13, 14) else "раундов"
    notice = f"Раунд {number} из {limit}. До обязательного голосования осталось {remaining} {unit}, включая текущий."
    if remaining == 1:
        return notice + " Это последний раунд переговоров; сразу после него начнётся голосование по достигнутым условиям."
    if remaining <= 3:
        return notice + " Время на согласование заканчивается; после последнего раунда голосование состоится даже при незакрытых вопросах."
    return notice + " После последнего раунда голосование состоится даже при незакрытых вопросах."


def run_game(config: GameConfig, chat: ChatModel, player_model: str, converter_model: str) -> dict[str, Any]:
    state = State({path: [] for path in config.required_paths})
    turns: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    session: list[dict[str, str]] = [
        {"kind": "opening", "text": "Слушания открыты. Участники: " + ", ".join(player.name for player in config.players) + ". Проект: " + "; ".join(f"{component.name} — {component.cost} ₽" for component in config.components) + f". Общая стоимость — {config.total_cost} ₽."},
        {"kind": "rules", "text": (
            "Оценка каждого участника: (желаемая доля собственного взноса × стоимость проекта − назначенный собственный взнос) / его бюджет "
            "+ баллы за ответственность. За каждый компонент на желаемом другом участнике начисляется +1, "
            "на себе −1; незаданное или конфликтное обязательство даёт 0. Незаданный или конфликтный взнос считается нулём для оценки. "
            "Взнос выше своего бюджета допустим: превышение можно покрыть кредитом, оно показано в таблице. Кредит входит во взнос участника; отдельный внешний кредит не покрывает стоимость проекта. "
            "Отрицательная оценка допустима и показывает цену взятых обязательств, но не запрещает сделку; сама по себе она не означает наличие кредита."
        )},
        {"kind": "table", "text": render_table(state, config)},
    ]
    seq = 1
    for round_number in range(1, config.max_rounds + 1):
        proposals: list[Assignment] = []
        speeches: list[dict[str, str]] = []
        session.append({"kind": "clock", "text": _round_notice(round_number, config.max_rounds)})
        for player in config.players:
            session.append({"kind": "floor", "text": f"Раунд {round_number}. Слово предоставлено: {player.name}."})
            speech = chat.complete(player_model, _player_messages(player, config, session))
            raw_sets, actions = convert_speech(chat, converter_model, speech, player.id, config, seq)
            seq += len(actions)
            proposals.extend(actions)
            speeches.append({"speaker": player.id, "speech": speech, "sets": raw_sets})
            session.append({"kind": "speech", "player": player.id, "text": f"{player.name}: {speech}"})
        # Atomic round commit: all conversion and validation finished above.
        state = merge_round(state, proposals, config)
        scores = score_state(state, config)
        snapshot = {"number": round_number, "speeches": speeches, "state": state.to_dict(), "table": render_table(state, config), "scores": scores, "funding_total": funding_total(state, config), "conflicts": state.conflicts()}
        rounds.append(snapshot)
        turns.extend(speeches)
        session.append({"kind": "table", "text": snapshot["table"]})
        if is_complete(state, config):
            break
    session.append({"kind": "stage", "text": "Голосование по договору открыто."})
    votes: dict[str, bool] = {}
    vote_actions: list[Assignment] = []
    for player in config.players:
        session.append({"kind": "floor", "text": f"Слово предоставлено: {player.name}."})
        speech, vote = _get_vote(chat, player_model, player, config, session)
        action = Assignment(f"participants.{player.id}.vote", vote, player.id, seq)
        seq += 1
        votes[player.id] = vote
        vote_actions.append(action)
        turns.append({"speaker": player.id, "speech": speech, "phase": "vote"})
        session.append({"kind": "vote", "player": player.id, "text": f"{player.name}: {speech}"})
    for action in vote_actions:
        state.fields[action.path] = [{"value": action.value, "player": action.player, "seq": action.seq}]
    accepted = sum(votes.values()) > len(config.players) / 2 and not state.conflicts() and funding_total(state, config) >= config.total_cost
    return {"total_cost": config.total_cost, "rounds": rounds, "final_state": state.to_dict(), "final_table": render_table(state, config), "final_scores": score_state(state, config), "votes": votes, "accepted": accepted, "turns": turns, "session": session}
