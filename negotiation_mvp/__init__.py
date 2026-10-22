"""Standalone negotiation game MVP."""

from .core import GameConfig, State, merge_round, render_table, score_state

__all__ = ["GameConfig", "State", "merge_round", "render_table", "score_state"]
