"""Game configuration for the messaging / alliance / observability layer.

`GameConfig` is the single knob bundle every game accepts via an optional
``config`` constructor argument. The no-arg constructor (``GameConfig()``)
reproduces the defaults, so existing games that pass nothing keep working
(INV-4).

See spec §3.1 / §8.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GameConfig:
    """Tunable behaviour for messaging, alliances, and steering A/B toggles.

    Attributes
    ----------
    messaging : bool
        Enable the negotiation / messaging sub-phase.
    alliances : bool
        Enable alliance proposals/accepts/declines/breaks.
    nego_rounds : int
        Negotiation rounds per negotiation phase (a game may override).
    msgs_per_slot : int
        Messages a seat may send per speaking slot.
    max_msg_chars : int
        Hard cap on a single message; longer text is truncated (never
        rejected) with a ``truncated`` flag so a verbose model cannot stall.
    enforce_alliances : bool
        Risk only: if True, alliance-violating attacks are illegal; if
        False (default) betrayal is physically possible and measurable.
    whisper_visibility : str
        ``"metadata"`` -> bystanders learn that a whisper happened (sender
        + recipient count, no text); ``"hidden"`` -> bystanders learn
        nothing at all.
    """

    messaging: bool = True
    alliances: bool = True
    nego_rounds: int = 2
    msgs_per_slot: int = 2
    max_msg_chars: int = 600
    enforce_alliances: bool = False
    whisper_visibility: str = "metadata"

    def as_dict(self) -> dict:
        """JSON-serialisable dict for the ``match_start.config`` record."""
        return asdict(self)
