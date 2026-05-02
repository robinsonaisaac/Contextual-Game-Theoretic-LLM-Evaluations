# game_theory_llm/games.py
"""Canonical 2x2 game definitions for multi-game experiments.

Each game is represented as a frozen ``GameConfig`` dataclass with:
- A ``PayoffMatrix`` encoding the 2x2 payoff structure
- Semantic labels for Decision A and Decision B (e.g., "Cooperate"/"Defect")
- Metadata about Nash equilibria, Pareto optimality, and game identity

The seven canonical games cover distinct strategic structures:

+---------------------+-------------+------------------+--------------------+
| Game                | Nash        | Key tension      | A-label / B-label  |
+---------------------+-------------+------------------+--------------------+
| Prisoner's Dilemma  | BB          | individual vs    | Cooperate / Defect |
|                     |             | collective       |                    |
| Stag Hunt           | AA, BB      | risk vs reward   | Hunt Stag /        |
|                     |             |                  | Hunt Hare          |
| Chicken (Hawk-Dove) | AB, BA      | bravado vs       | Swerve / Dare      |
|                     |             | caution          |                    |
| Deadlock            | BB          | defect dominant  | Cooperate / Defect |
|                     |             | & Pareto-best    |                    |
| Harmony             | AA          | none (trivially  | Cooperate / Defect |
|                     |             | rational)        |                    |
| Battle of the Sexes | AA, BB      | asymmetric       | Plan Alpha /       |
|                     |             | coordination     | Plan Beta          |
| Matching Pennies    | (mixed)     | zero-sum         | Heads / Tails      |
|                     |             | opposition       |                    |
+---------------------+-------------+------------------+--------------------+
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .models import PayoffMatrix


@dataclass(frozen=True)
class GameConfig:
    """Immutable configuration for a 2x2 game.

    Parameters
    ----------
    id : str
        Machine-readable identifier (e.g. ``"prisoners_dilemma"``).
    name : str
        Human-readable name for display.
    matrix : PayoffMatrix
        The 2x2 payoff matrix (AA, AB, BA, BB).
    label_a : str
        Semantic label for Decision A (the "focal" cooperative choice).
    label_b : str
        Semantic label for Decision B.
    focal_decision : str
        Which decision is the "cooperative" one for analysis (always ``"A"``
        in canonical form).
    nash_equilibria : tuple
        Nash equilibrium profiles (e.g. ``("BB",)`` for PD).
    pareto_optimal : str
        The Pareto-optimal outcome profile.
    description : str
        One-line description of the game's strategic structure.
    framing_hint : str
        Single sentence injected into the story-generation prompt to steer the
        LLM toward the game's strategic structure. Should reference Decision A
        and Decision B explicitly.
    """
    id: str
    name: str
    matrix: PayoffMatrix
    label_a: str
    label_b: str
    focal_decision: str = "A"
    nash_equilibria: Tuple[str, ...] = ()
    pareto_optimal: str = "AA"
    description: str = ""
    framing_hint: str = ""


# ---------------------------------------------------------------------------
# Canonical game definitions
# ---------------------------------------------------------------------------
# Payoff orderings follow standard game-theory textbook conventions.
# Matrix order: (AA, AB, BA, BB) where first element = Agent 1's payoff.

_PRISONERS_DILEMMA = GameConfig(
    id="prisoners_dilemma",
    name="Prisoner's Dilemma",
    matrix=PayoffMatrix([(3, 3), (0, 5), (5, 0), (1, 1)]),
    label_a="Cooperate",
    label_b="Defect",
    nash_equilibria=("BB",),
    pareto_optimal="AA",
    description="T>R>P>S: dominant strategy to defect, but mutual cooperation is Pareto-optimal",
    framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice.",
)

_STAG_HUNT = GameConfig(
    id="stag_hunt",
    name="Stag Hunt",
    matrix=PayoffMatrix([(4, 4), (0, 3), (3, 0), (2, 2)]),
    label_a="Hunt Stag",
    label_b="Hunt Hare",
    nash_equilibria=("AA", "BB"),
    pareto_optimal="AA",
    description="R>T>P>S: two pure Nash equilibria, payoff-dominant vs risk-dominant",
    framing_hint="Decision A is the high-reward joint action requiring mutual commitment; Decision B is the safer individual fallback.",
)

_CHICKEN = GameConfig(
    id="chicken",
    name="Chicken (Hawk-Dove)",
    matrix=PayoffMatrix([(3, 3), (1, 4), (4, 1), (0, 0)]),
    label_a="Swerve",
    label_b="Dare",
    nash_equilibria=("AB", "BA"),
    pareto_optimal="AA",
    description="T>R>S>P: anti-coordination game with two asymmetric Nash equilibria",
    framing_hint="Decision A is the yielding/cautious action; Decision B is the assertive/confrontational action.",
)

_DEADLOCK = GameConfig(
    id="deadlock",
    name="Deadlock",
    matrix=PayoffMatrix([(1, 1), (0, 3), (3, 0), (2, 2)]),
    label_a="Cooperate",
    label_b="Defect",
    nash_equilibria=("BB",),
    pareto_optimal="BB",
    description="T>P>R>S: defection dominates AND is mutually preferred — no dilemma (PD's control)",
    framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice.",
)

_HARMONY = GameConfig(
    id="harmony",
    name="Harmony",
    matrix=PayoffMatrix([(4, 4), (2, 3), (3, 2), (1, 1)]),
    label_a="Cooperate",
    label_b="Defect",
    nash_equilibria=("AA",),
    pareto_optimal="AA",
    description="R>T>S>P: cooperation is the dominant strategy — positive control for PD",
    framing_hint="Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice.",
)

_BATTLE_OF_THE_SEXES = GameConfig(
    id="battle_of_the_sexes",
    name="Battle of the Sexes",
    matrix=PayoffMatrix([(3, 2), (0, 0), (0, 0), (2, 3)]),
    label_a="Plan Alpha",
    label_b="Plan Beta",
    nash_equilibria=("AA", "BB"),
    pareto_optimal="AA",
    description="Asymmetric coordination: both prefer to coordinate but disagree on which outcome",
    framing_hint="Decision A and Decision B both represent coordination options, but each agent prefers a different one. Failing to coordinate is the worst outcome for both.",
)

_MATCHING_PENNIES = GameConfig(
    id="matching_pennies",
    name="Matching Pennies",
    matrix=PayoffMatrix([(1, 0), (0, 1), (0, 1), (1, 0)]),
    label_a="Heads",
    label_b="Tails",
    nash_equilibria=(),  # only mixed-strategy NE at (0.5, 0.5)
    pareto_optimal="AA",
    description="Zero-sum: player 1 wants to match, player 2 wants to mismatch — no pure NE",
    framing_hint="The two decisions are arbitrary symbolic choices in an adversarial encounter — Decision A and Decision B have no inherent meaning, but one agent benefits when both choose alike and the other benefits when they choose differently.",
)


# ---------------------------------------------------------------------------
# Registry and helpers
# ---------------------------------------------------------------------------

GAME_REGISTRY: Dict[str, GameConfig] = {
    g.id: g
    for g in (
        _PRISONERS_DILEMMA,
        _STAG_HUNT,
        _CHICKEN,
        _DEADLOCK,
        _HARMONY,
        _BATTLE_OF_THE_SEXES,
        _MATCHING_PENNIES,
    )
}

ALL_GAME_IDS: List[str] = list(GAME_REGISTRY.keys())

DEFAULT_GAME: GameConfig = _PRISONERS_DILEMMA


def get_game(game_id: str) -> GameConfig:
    """Return the ``GameConfig`` for *game_id*.

    Raises ``ValueError`` if the game ID is not in the registry.
    """
    if game_id not in GAME_REGISTRY:
        raise ValueError(
            f"Unknown game {game_id!r}. Choose from: {ALL_GAME_IDS}"
        )
    return GAME_REGISTRY[game_id]
