"""Funding and score calculations for the negotiation game."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from negotiation_mvp.core import GameConfig, State


def funding_total(state: State, config: GameConfig) -> int:
    return sum(state.value(f"participants.{player.id}.contribution") or 0 for player in config.players)


def score_state(state: State, config: GameConfig, *, signed: bool = False) -> dict[str, float]:
    """Оценка 0–100; комментарии ведут один пример через каждую строку.

    Состояние примера: проект 100, два игрока с бюджетами 50 и 50,
    взносами 40 и 40; рассматриваем игрока A с желаемой долей 0.5.
    Один компонент назначен желаемому для A игроку B, договор не принят.
    """
    # Начало: оценок ещё нет -> {}.
    scores = {}
    # Бюджеты 50 и 50 -> total_budget = 100.
    total_budget = sum(player.budget for player in config.players)
    # Взносы 40 + 40 = 80 -> funding_gap = (80 - 100) / 100 = -0.2.
    funding_gap = (min(funding_total(state, config), config.total_cost) - config.total_cost) / (total_budget or config.total_cost)
    # Первая итерация: player = A; после неё scores = {"A": 60}.
    for player in config.players:
        # Для A: fact = 40; отсутствующий взнос считался бы нулём.
        fact = state.value(f"participants.{player.id}.contribution") or 0
        # Для A: (0.5 * 100 - 40) / 50 - 0.2 = 0 -> score = 0.
        score = ((player.expected_own_share * config.total_cost - fact) / player.budget
                 if player.budget else player.expected_own_share - fact / config.total_cost) + funding_gap
        # Для A: одна итерация, component_id = выбранный компонент, desired = B.
        for component_id, desired in player.desired_responsible.items():
            # Для A: actual = B, потому что компонент назначен B.
            actual = state.value(f"components.{component_id}.responsible")
            # В примере actual задан, поэтому перехода к следующему компоненту нет.
            if actual is None:
                continue
            # actual != A, поэтому ветка -1 пропускается; score остаётся 0.
            if actual == player.id:
                score -= 1
            # actual == desired == B -> score увеличивается с 0 до 1.
            elif actual == desired:
                score += 1
        # signed=False -> 100 + 10 * 1 - 50 = 60 -> scores["A"] = 60.
        scores[player.id] = round(max(0, min(100, 100 + 10 * score - (0 if signed else 50))), 10)
    # После итераций словарь содержит оценки обоих игроков; возвращаем его.
    return scores
