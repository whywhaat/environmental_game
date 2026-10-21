"""Deterministic configuration, SET validation, merge, scoring and rendering."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass
from typing import Any


class GameError(ValueError):
    """Invalid game configuration or command."""


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    budget: int
    expected_own_share: float
    desired_responsible: dict[str, str]
    personal_context: str
    motivation: str
    personal_history: tuple[str, ...]


@dataclass(frozen=True)
class Component:
    id: str
    name: str
    cost: int


@dataclass(frozen=True)
class GameConfig:
    players: tuple[Player, ...]
    components: tuple[Component, ...]
    max_rounds: int = 3

    @property
    def total_cost(self) -> int:
        return sum(component.cost for component in self.components)

    @property
    def player_ids(self) -> set[str]:
        return {player.id for player in self.players}

    @property
    def component_ids(self) -> set[str]:
        return {component.id for component in self.components}

    @property
    def required_paths(self) -> tuple[str, ...]:
        return tuple(
            [f"participants.{player.id}.contribution" for player in self.players]
            + [f"components.{component.id}.responsible" for component in self.components]
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GameConfig:
        if not isinstance(raw, dict):
            raise GameError("Configuration must be a JSON object")
        raw_players = raw.get("players")
        raw_components = raw.get("components")
        if not isinstance(raw_players, list) or len(raw_players) < 2:
            raise GameError("At least two players are required")
        if not isinstance(raw_components, list) or not raw_components:
            raise GameError("At least one cost component is required")
        components = []
        for item in raw_components:
            if not isinstance(item, dict):
                raise GameError("Each component must be an object")
            ident = _identifier(item.get("id"), "component id")
            components.append(Component(ident, _nonempty_text(item.get("name"), "component name"), _whole(item.get("cost"), "component cost", minimum=0)))
        if len({item.id for item in components}) != len(components):
            raise GameError("Component ids must be unique")
        players = []
        for item in raw_players:
            if not isinstance(item, dict):
                raise GameError("Each player must be an object")
            ident = _identifier(item.get("id"), "player id")
            share = item.get("expected_own_share")
            if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 <= share <= 1:
                raise GameError(f"Invalid expected_own_share for {ident}")
            desired = item.get("desired_responsible", {})
            if not isinstance(desired, dict):
                raise GameError(f"desired_responsible for {ident} must be an object")
            personal_history = item.get("personal_history")
            if not isinstance(personal_history, list) or not personal_history:
                raise GameError(f"personal_history for {ident} must be a nonempty list")
            players.append(Player(
                ident,
                _nonempty_text(item.get("name"), "player name"),
                _whole(item.get("budget"), "player budget", minimum=0),
                float(share),
                desired,
                _nonempty_text(item.get("personal_context"), "personal context"),
                _nonempty_text(item.get("motivation"), "motivation"),
                tuple(_nonempty_text(event, "personal history event") for event in personal_history),
            ))
        ids = {item.id for item in players}
        if len(ids) != len(players):
            raise GameError("Player ids must be unique")
        if len({item.name for item in players}) != len(players):
            raise GameError("Player names must be unique")
        obligation_ids = {item.id for item in components}
        for player in players:
            if set(player.desired_responsible) != obligation_ids:
                raise GameError(f"desired_responsible for {player.id} must list every component")
            if any(owner not in ids or owner == player.id for owner in player.desired_responsible.values()):
                raise GameError(f"{player.id} must prefer another configured player for each obligation")
        rounds = _whole(raw.get("max_rounds", 3), "max_rounds", minimum=1)
        config = cls(tuple(players), tuple(components), rounds)
        if config.total_cost <= 0:
            raise GameError("Total component cost must be positive")
        return config


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", value):
        raise GameError(f"Invalid {label}: {value!r}")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GameError(f"Invalid {label}")
    return value.strip()


def _whole(value: Any, label: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GameError(f"Invalid {label}: {value!r}")
    return value


@dataclass(frozen=True)
class Assignment:
    path: str
    value: int | str | bool
    player: str
    seq: int


class State:
    """Current field values and their attribution, without a turn history."""

    def __init__(self, fields: dict[str, list[dict[str, Any]]] | None = None):
        self.fields = copy.deepcopy(fields or {})

    def to_dict(self) -> dict[str, Any]:
        return {"fields": copy.deepcopy(self.fields)}

    def records(self, path: str) -> list[dict[str, Any]]:
        return self.fields.get(path, [])

    def conflict(self, path: str) -> bool:
        return len({json.dumps(item["value"], ensure_ascii=False, sort_keys=True) for item in self.records(path)}) > 1

    def value(self, path: str) -> Any | None:
        records = self.records(path)
        return records[0]["value"] if records and not self.conflict(path) else None

    def conflicts(self) -> list[str]:
        return [path for path in self.fields if self.conflict(path)]


_SET_RE = re.compile(r"^SET\s+([a-z][a-z0-9_.]*)\s*=\s*(.+?)\s*$", re.ASCII)


def parse_sets(text: str, speaker: str, config: GameConfig, start_seq: int = 1) -> list[Assignment]:
    if speaker not in config.player_ids:
        raise GameError(f"Unknown speaker: {speaker}")
    if not isinstance(text, str):
        raise GameError("SET output must be text")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines == ["NO_CHANGES"] or not lines:
        return []
    result: list[Assignment] = []
    seen: set[str] = set()
    for line in lines:
        match = _SET_RE.fullmatch(line)
        if not match:
            raise GameError(f"Invalid SET line: {line}")
        path, value_text = match.groups()
        if path in seen:
            raise GameError(f"Duplicate SET field in one speech: {path}")
        seen.add(path)
        if path not in config.required_paths and not re.fullmatch(r"participants\.[a-z][a-z0-9_]*\.vote", path):
            raise GameError(f"Unknown SET path: {path}")
        parts = path.split(".")
        if parts[0] == "participants" and parts[1] not in config.player_ids:
            raise GameError(f"Unknown participant in SET: {parts[1]}")
        try:
            value = json.loads(value_text)
        except json.JSONDecodeError as exc:
            raise GameError(f"Invalid JSON value for {path}") from exc
        if path.endswith(".contribution"):
            value = _whole(value, path, minimum=0)
        elif path.endswith(".responsible"):
            if not isinstance(value, str) or value not in config.player_ids:
                raise GameError(f"Invalid responsible player for {path}")
        elif path.endswith(".vote"):
            if not isinstance(value, bool) or parts[1] != speaker:
                raise GameError("A player may set only their own boolean vote")
        result.append(Assignment(path, value, speaker, start_seq + len(result)))
    return result


def merge_round(state: State, assignments: list[Assignment], config: GameConfig) -> State:
    next_state = State(state.fields)
    for path in list(next_state.fields):
        if next_state.conflict(path):
            next_state.fields[path] = []
    for item in assignments:
        if item.path not in config.required_paths:
            raise GameError(f"Not a negotiation field: {item.path}")
        records = next_state.fields.setdefault(item.path, [])
        # A repeated endorsement replaces that player's earlier identical record.
        records[:] = [record for record in records if not (record["player"] == item.player and record["value"] == item.value)]
        records.append({"value": item.value, "player": item.player, "seq": item.seq})
    return next_state


def score_state(state: State, config: GameConfig) -> dict[str, float]:
    scores = {}
    for player in config.players:
        fact = state.value(f"participants.{player.id}.contribution") or 0
        score = ((player.expected_own_share * config.total_cost - fact) / player.budget
                 if player.budget else player.expected_own_share - fact / config.total_cost)
        for component_id, desired in player.desired_responsible.items():
            actual = state.value(f"components.{component_id}.responsible")
            if actual is None:
                continue
            if actual == player.id:
                score -= 1
            elif actual == desired:
                score += 1
        scores[player.id] = round(score, 10)
    return scores


def funding_total(state: State, config: GameConfig) -> int:
    return sum(state.value(f"participants.{player.id}.contribution") or 0 for player in config.players)


def is_complete(state: State, config: GameConfig) -> bool:
    return not state.conflicts() and all(state.value(path) is not None for path in config.required_paths) and funding_total(state, config) >= config.total_cost


def render_table(state: State, config: GameConfig) -> str:
    names = {player.id: player.name for player in config.players}
    component_names = {component.id: component.name for component in config.components}
    rows = ["| Поле | Значение и автор | Конфликт |", "| --- | --- | --- |"]
    paths = list(config.required_paths) + [f"participants.{player.id}.vote" for player in config.players if state.records(f"participants.{player.id}.vote")]
    for path in paths:
        label = path
        if path.startswith("participants."):
            player_id = path.split(".")[1]
            label = f"Голос: {names[player_id]}" if path.endswith(".vote") else f"Взнос: {names[player_id]}"
        elif path.endswith(".responsible"):
            component = path.split(".")[1]
            label = f"Ответственный: {component_names[component]}"
        records = sorted(state.records(path), key=lambda record: record["seq"])
        displayed = []
        for record in records:
            value = record["value"]
            if path.endswith(".contribution"):
                value = f"{value / 1_000_000:g} млн ₽"
            elif path.endswith(".responsible"):
                value = names[value]
            elif path.endswith(".vote"):
                value = "За" if value else "Против"
            displayed.append(f"{value} ({names[record['player']]})")
        rows.append(f"| {label} | {'; '.join(displayed) if displayed else '—'} | {'Да' if state.conflict(path) else 'Нет'} |")
    scores = score_state(state, config)
    for player in config.players:
        score = f"{scores[player.id]:+g}".replace(".", ",")
        contribution = state.value(f"participants.{player.id}.contribution")
        credit = "—" if contribution is None else f"{max(0, contribution - player.budget) / 1_000_000:g} млн ₽"
        rows.append(f"| Оценка: {player.name} | {score} | — |")
        rows.append(f"| Кредит: {player.name} | {credit} | — |")
    return "\n".join(rows)
