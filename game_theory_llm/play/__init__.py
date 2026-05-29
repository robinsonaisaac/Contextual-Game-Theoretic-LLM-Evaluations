"""Multi-game play harness for LLM-as-player evaluation.

See ``docs/game_play_harness.md`` for design. Goal: take an activation
steering vector fit on the vignette corpora and measure its causal effect
on the model's behaviour as a player in real strategic games.
"""

from .base import Action, Game, Player, MatchResult
from .runner import run_match

__all__ = ["Action", "Game", "Player", "MatchResult", "run_match"]
