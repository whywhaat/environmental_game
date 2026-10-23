"""Deterministic configuration, SET validation, merge, scoring and rendering."""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any


class GameError(ValueError):
    """Invalid game configuration or command."""


LEVER_SPECS: dict[str, dict[str, Any]] = {
    "guilt": {
        "type": str,
        "allowed": {"shared", "holding", "city"},
        "default": "shared",
        "label": "Вина за аварию",
        "owner": "regional_government",
    },
    "plant_mode": {
        "type": str,
        "allowed": {"100%", "70%"},
        "default": "100%",
        "label": "Режим завода",
        "owner": "industrial_holding",
    },
    "timeline_months": {
        "type": int,
        "allowed": {24, 36, 48},
        "default": 36,
        "label": "Срок реконструкции (мес.)",
        "owner": "industrial_holding",
    },
    "tariffs_freeze": {
        "type": bool,
        "allowed": {True, False},
        "default": False,
        "label": "Заморозка тарифов ЖКХ",
        "owner": "city_administration",
    },
    "tax_relief": {
        "type": bool,
        "allowed": {True, False},
        "default": False,
        "label": "Налоговые льготы заводу",
        "owner": "regional_government",
    },
    "open_sensors": {
        "type": bool,
        "allowed": {True, False},
        "default": False,
        "label": "Открытые эко-датчики 24/7",
        "owner": "environmental_ngo",
    },
    "protest_moratorium": {
        "type": bool,
        "allowed": {True, False},
        "default": False,
        "label": "Мораторий на протесты и иски",
        "owner": "city_residents",
    },
}



@dataclass(frozen=True)
class Player:
    id: str
    name: str
    budget: int
    expected_own_share: float = 0.0
    desired_responsible: dict[str, str] = field(default_factory=dict)
    personal_context: str = ""
    motivation: str = ""
    personal_history: tuple[str, ...] = ()
    prompt: str = ""


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
    def parameter_paths(self) -> tuple[str, ...]:
        return tuple(f"parameters.{key}" for key in LEVER_SPECS)

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
        all_ids = [_identifier(item.get("id"), "player id") for item in raw_players if isinstance(item, dict) and item.get("id")]
        obligation_ids = {item.id for item in components}
        for item in raw_players:
            if not isinstance(item, dict):
                raise GameError("Each player must be an object")
            ident = _identifier(item.get("id"), "player id")
            share = item.get("expected_own_share")
            if share is None or share == "":
                share = 0.0
            if isinstance(share, bool) or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 <= share <= 1:
                raise GameError(f"Invalid expected_own_share for {ident}")
            desired_raw = item.get("desired_responsible")
            if desired_raw is None or desired_raw == "":
                desired_raw = {}
            if not isinstance(desired_raw, dict):
                raise GameError(f"desired_responsible for {ident} must be an object")
            desired = dict(desired_raw)
            for comp_id in obligation_ids:
                if comp_id not in desired:
                    fallback = next((pid for pid in all_ids if pid != ident), ident)
                    desired[comp_id] = fallback

            personal_history = item.get("personal_history")
            if personal_history is None or personal_history == "":
                personal_history = []
            if not isinstance(personal_history, (list, tuple)):
                raise GameError(f"personal_history for {ident} must be a list")

            prompt = str(item.get("prompt") or "").strip()
            raw_context = item.get("personal_context")
            if raw_context is None or not str(raw_context).strip():
                context = prompt or f"Вы представляете {item.get('name', ident)} в переговорах."
            else:
                context = str(raw_context).strip()

            raw_motivation = item.get("motivation")
            if raw_motivation is None or not str(raw_motivation).strip():
                motivation = "Добейтесь заключения договора в интересах своей стороны."
            else:
                motivation = str(raw_motivation).strip()

            players.append(Player(
                ident,
                _nonempty_text(item.get("name"), "player name"),
                _whole(item.get("budget"), "player budget", minimum=0),
                float(share),
                desired,
                context,
                motivation,
                tuple(_nonempty_text(event, "personal history event") for event in personal_history if str(event).strip()),
                prompt,
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
            if any(owner not in ids for owner in player.desired_responsible.values()):
                raise GameError(f"Unknown player in desired_responsible for {player.id}")
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
        is_param = path.startswith("parameters.")
        is_vote = bool(re.fullmatch(r"participants\.[a-z][a-z0-9_]*\.vote", path))
        if path not in config.required_paths and not is_param and not is_vote:
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
        elif is_param:
            param_key = path.split(".", 1)[1]
            if param_key not in LEVER_SPECS:
                raise GameError(f"Unknown parameter: {param_key}")
            spec = LEVER_SPECS[param_key]
            if spec["type"] is int and isinstance(value, str) and value.isdigit():
                value = int(value)
            elif spec["type"] is bool and isinstance(value, str):
                if value.lower() in ("true", "да", "1"):
                    value = True
                elif value.lower() in ("false", "нет", "0"):
                    value = False
            if value not in spec["allowed"]:
                raise GameError(f"Invalid value {value!r} for parameter {param_key}, allowed: {sorted(spec['allowed'], key=str)}")
        result.append(Assignment(path, value, speaker, start_seq + len(result)))
    return result


def merge_round(state: State, assignments: list[Assignment], config: GameConfig) -> State:
    next_state = State(state.fields)
    valid_paths = set(config.required_paths) | set(config.parameter_paths)
    for item in assignments:
        if item.path not in valid_paths:
            raise GameError(f"Not a negotiation field: {item.path}")
        records = next_state.fields.setdefault(item.path, [])
        records[:] = [record for record in records if record["player"] != item.player]
        records.append({"value": item.value, "player": item.player, "seq": item.seq})
    return next_state


def score_state(state: State, config: GameConfig, *, signed: bool = False) -> dict[str, float]:
    scores = {}
    total_budget = sum(player.budget for player in config.players)
    funding_gap = (min(funding_total(state, config), config.total_cost) - config.total_cost) / (total_budget or config.total_cost)
    params = {k: state.value(f"parameters.{k}") for k in LEVER_SPECS}

    for player in config.players:
        fact = state.value(f"participants.{player.id}.contribution") or 0
        score = ((player.expected_own_share * config.total_cost - fact) / player.budget
                 if player.budget else player.expected_own_share - fact / config.total_cost) + funding_gap
        for component_id, desired in player.desired_responsible.items():
            actual = state.value(f"components.{component_id}.responsible")
            if actual is None:
                continue
            if actual == desired:
                score += 1
            elif actual == player.id:
                score -= 1

        pid = player.id
        guilt = params.get("guilt")
        if guilt == "shared":
            if pid in ("industrial_holding", "city_administration", "regional_government"):
                score += 1.0
            elif pid == "city_residents":
                score += 0.5
        elif guilt == "holding":
            if pid == "industrial_holding":
                score -= 2.0
            elif pid in ("city_administration", "city_residents", "environmental_ngo"):
                score += 1.5
        elif guilt == "city":
            if pid == "city_administration":
                score -= 2.0
            elif pid == "industrial_holding":
                score += 1.5

        plant = params.get("plant_mode")
        if plant == "100%":
            if pid == "industrial_holding":
                score += 1.5
            elif pid in ("city_residents", "environmental_ngo"):
                score -= 1.0
        elif plant == "70%":
            if pid in ("city_residents", "environmental_ngo"):
                score += 1.5
            elif pid == "industrial_holding":
                score -= 0.5

        tariffs = params.get("tariffs_freeze")
        if tariffs is True:
            if pid == "city_residents":
                score += 2.0
            elif pid == "city_administration":
                score += 0.5
        elif tariffs is False and "parameters.tariffs_freeze" in state.fields:
            if pid == "city_residents":
                score -= 1.5

        timeline = params.get("timeline_months")
        if timeline == 24:
            if pid in ("city_residents", "city_administration"):
                score += 1.5
            elif pid == "industrial_holding":
                score -= 0.5
        elif timeline == 48:
            if pid == "industrial_holding":
                score += 1.0
            elif pid in ("city_residents", "city_administration"):
                score -= 1.0
        elif timeline == 36:
            if pid in ("city_residents", "city_administration", "industrial_holding"):
                score += 0.5

        tax = params.get("tax_relief")
        if tax is True:
            if pid == "industrial_holding":
                score += 1.0
            elif pid == "regional_government":
                score -= 0.5

        sensors = params.get("open_sensors")
        if sensors is True:
            if pid == "environmental_ngo":
                score += 2.0
            elif pid == "city_residents":
                score += 0.5

        protest = params.get("protest_moratorium")
        if protest is True:
            if pid in ("city_administration", "regional_government", "industrial_holding"):
                score += 0.5

        scores[player.id] = round(max(0, min(100, 100 + 10 * score - (0 if signed else 50))), 10)
    return scores


def funding_total(state: State, config: GameConfig) -> int:
    return sum(state.value(f"participants.{player.id}.contribution") or 0 for player in config.players)


def is_complete(state: State, config: GameConfig) -> bool:
    return not state.conflicts() and all(state.value(path) is not None for path in config.required_paths) and funding_total(state, config) >= config.total_cost


def render_table(state: State, config: GameConfig, *, signed: bool = False) -> str:
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
    param_paths = [p for p in config.parameter_paths if state.records(p)]
    for path in param_paths:
        param_key = path.split(".", 1)[1]
        spec = LEVER_SPECS.get(param_key, {})
        label = f"Условие: {spec.get('label', param_key)}"
        records = sorted(state.records(path), key=lambda r: r["seq"])
        displayed = []
        for record in records:
            val = record["value"]
            val_str = "Да" if val is True else ("Нет" if val is False else str(val))
            displayed.append(f"{val_str} ({names.get(record['player'], record['player'])})")
        rows.append(f"| {label} | {'; '.join(displayed) if displayed else '—'} | {'Да' if state.conflict(path) else 'Нет'} |")
    scores = score_state(state, config, signed=signed)
    for player in config.players:
        score = f"{scores[player.id]:g}".replace(".", ",")
        contribution = state.value(f"participants.{player.id}.contribution")
        credit = "—" if contribution is None else f"{max(0, contribution - player.budget) / 1_000_000:g} млн ₽"
        rows.append(f"| Оценка: {player.name} | {score} | — |")
        rows.append(f"| Кредит: {player.name} | {credit} | — |")
    return "\n".join(rows)
