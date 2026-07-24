"""Multi-game play harness for LLM-as-player evaluation.

See ``docs/game_play_harness.md`` and ``docs/play_harness_v2_spec.md`` for
design. Goal: take an activation steering vector fit on the vignette corpora
and measure its causal effect on the model's behaviour as a player in real
strategic games — including how it negotiates, forms alliances, and betrays.
"""

from .base import GOD, Action, Game, MatchResult, Obs, Observation, Player
from .config import GameConfig
from .messaging import MessagingMixin, NegotiationState
from .alliances import (
    Alliance,
    AllianceMixin,
    AllianceState,
    alliance_summary,
    new_event,
)
from .runner import run_match

__all__ = [
    "Action",
    "Game",
    "Player",
    "MatchResult",
    "Obs",
    "Observation",
    "GOD",
    "GameConfig",
    "MessagingMixin",
    "NegotiationState",
    "AllianceMixin",
    "AllianceState",
    "Alliance",
    "alliance_summary",
    "new_event",
    "run_match",
]
