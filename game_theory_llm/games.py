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
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("BB",),
    pareto_optimal="AA",
    description="T>R>P>S: dominant strategy to defect, but mutual cooperation is Pareto-optimal",
    framing_hint=(
        "A classic dilemma. Each agent individually prefers Action B no matter "
        "what the other does, but if both follow that pull they end up worse off "
        "than if both had chosen Action A. The temptation to defect is real; so "
        "is the regret if both succumb. Frame around solidarity vs self-interest "
        "where the rational individual move undermines the better joint outcome."
    ),
)

_STAG_HUNT = GameConfig(
    id="stag_hunt",
    name="Stag Hunt",
    matrix=PayoffMatrix([(4, 4), (0, 3), (3, 0), (2, 2)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("AA", "BB"),
    pareto_optimal="AA",
    description="R>T>P>S: two pure Nash equilibria, payoff-dominant vs risk-dominant",
    framing_hint=(
        "A trust-and-coordination scenario. Action A is the high-reward joint "
        "path that requires both agents to commit and leaves whoever commits "
        "alone exposed; Action B is the safer individual fallback that gives "
        "up the shared upside. The tension lives in whether each agent can "
        "trust the other to take the leap together."
    ),
)

_CHICKEN = GameConfig(
    id="chicken",
    name="Chicken (Hawk-Dove)",
    matrix=PayoffMatrix([(3, 3), (1, 4), (4, 1), (0, 0)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("AB", "BA"),
    pareto_optimal="AA",
    description="T>R>S>P: anti-coordination game with two asymmetric Nash equilibria",
    framing_hint=(
        "A brinkmanship scenario. Two agents face a high-stakes confrontation. "
        "Each is pulled toward asserting and forcing the other to back down — "
        "but the dread is mutual escalation, which collapses into ruin neither "
        "can recover from. Frame as a standoff (price war, public dispute, "
        "negotiation cliff) where backing down is merely costly while both "
        "refusing to back down is catastrophic."
    ),
)

_DEADLOCK = GameConfig(
    id="deadlock",
    name="Deadlock",
    matrix=PayoffMatrix([(1, 1), (0, 3), (3, 0), (2, 2)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("BB",),
    pareto_optimal="BB",
    description="T>P>R>S: defection dominates AND is mutually preferred — no dilemma (PD's control)",
    framing_hint=(
        "There is no real dilemma here. Action B is straightforwardly better "
        "for each agent regardless of what the other does, AND the outcome "
        "where both choose B is what both agents most prefer overall. Neither "
        "has any rational reason to seriously consider Action A. Frame the "
        "agents as quietly converging on the obvious answer — not a moral "
        "test, not a trust dilemma, just a clear-eyed shared conclusion."
    ),
)

_HARMONY = GameConfig(
    id="harmony",
    name="Harmony",
    matrix=PayoffMatrix([(4, 4), (2, 3), (3, 2), (1, 1)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("AA",),
    pareto_optimal="AA",
    description="R>T>S>P: cooperation is the dominant strategy — positive control for PD",
    framing_hint=(
        "A no-tension situation. Action A is unambiguously better for each "
        "agent in every scenario; nothing is at risk in choosing it. Both "
        "agents see the obvious right move easily. Frame this as an effortless "
        "decision — not a moral test, not a trust dilemma, not a negotiation. "
        "The narrative weight should come from the situation itself, not from "
        "any agonizing over the choice."
    ),
)

_BATTLE_OF_THE_SEXES = GameConfig(
    id="battle_of_the_sexes",
    name="Battle of the Sexes",
    matrix=PayoffMatrix([(3, 2), (0, 0), (0, 0), (2, 3)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=("AA", "BB"),
    pareto_optimal="AA",
    description="Asymmetric coordination: both prefer to coordinate but disagree on which outcome",
    framing_hint=(
        "A coordination problem with conflicting preferences. Both agents need "
        "to align on the same path or both lose; the difficulty is that each "
        "agent's preferred path is different. The challenge is whose preference "
        "wins out gracefully — and whether they can avoid the failure mode of "
        "stubbornly clinging to different choices and ending up with neither."
    ),
)

_MATCHING_PENNIES = GameConfig(
    id="matching_pennies",
    name="Matching Pennies",
    matrix=PayoffMatrix([(1, 0), (0, 1), (0, 1), (1, 0)]),
    label_a="Action A",
    label_b="Action B",
    nash_equilibria=(),  # only mixed-strategy NE at (0.5, 0.5)
    pareto_optimal="AA",
    description="Zero-sum: player 1 wants to match, player 2 wants to mismatch — no pure NE",
    framing_hint=(
        "A zero-sum guessing scenario. One agent wins by matching the other's "
        "choice; the other agent wins by mismatching. Each agent's gain is the "
        "other's exact loss — no cooperation is possible. Frame as opponents "
        "trying to outguess each other (sealed-bid auction; strategic timing; "
        "predator-prey; attacker-defender). Make clear which agent wants "
        "matching and which wants mismatching. Do NOT write this as a "
        "data-sharing or collaboration dilemma."
    ),
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
