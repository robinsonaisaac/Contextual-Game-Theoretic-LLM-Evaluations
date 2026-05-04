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
        "Both agents have a dominant incentive to choose Action B: against the "
        "other's A it pays 5 vs 3, and against the other's B it pays 1 vs 0. "
        "So both will rationally end up at (1,1) — even though mutual A would "
        "have given (3,3). The strategic tension is between individual rationality "
        "(each agent's best response is B) and joint welfare (both prefer mutual A)."
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
        "Two equilibria. Mutual A gives the best joint outcome (4,4); mutual B "
        "is the safe fallback (2,2). But choosing A unilaterally against the "
        "other's B gives 0 — the worst individual outcome. This is a coordination "
        "problem: Action A is the high-reward joint action requiring mutual "
        "commitment; Action B is the safer individual fallback. The tension is "
        "between payoff-dominance (A) and risk-dominance (B)."
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
        "Mutual B is CATASTROPHIC for both (0,0) — the worst outcome. "
        "Mutual A is acceptable (3,3). Each agent prefers to choose B against "
        "the other's A (4 vs 3), but absolutely wants to avoid mutual B. "
        "The two equilibria are off-diagonal: one agent picks A while the "
        "other picks B (the agent picking B gets 4, the one picking A gets 1). "
        "This is brinkmanship — each agent wants to assert by choosing B, "
        "but neither wants mutual confrontation. Frame the scenario as a "
        "high-stakes confrontation (price war, standoff, negotiation cliff, "
        "public dispute) where backing down (A) is merely costly but mutual "
        "escalation (B) is ruinous."
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
        "Both agents have a dominant incentive to choose Action B: against the "
        "other's A it pays 3 vs 1, and against the other's B it pays 2 vs 0. "
        "Mutual B (2,2) is ALSO mutually preferred to mutual A (1,1). Choosing "
        "B is BOTH individually rational AND collectively optimal — there is "
        "NO dilemma here. Both agents should clearly prefer B; they have little "
        "reason to seriously consider A. Do NOT portray this as a difficult "
        "moral dilemma or a test of trust — frame it as a straightforward "
        "decision where the agents naturally converge on B."
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
        "Action A strictly dominates Action B for both agents: against the "
        "other's A it pays 4 vs 3, and against the other's B it pays 2 vs 1. "
        "Mutual A (4,4) is also Pareto-optimal. There is NO strategic tension — "
        "A is the obvious right choice for both agents regardless of what the "
        "other does. Frame this as an easy decision with no real dilemma; "
        "neither agent has reason to consider B."
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
        "Both agents want to coordinate (mutual A or mutual B), but they "
        "disagree on which: agent 1 prefers mutual A (3,2), agent 2 prefers "
        "mutual B (2,3). Mismatched choices (AB or BA) give 0 to both — the "
        "worst outcome. The challenge is choosing between two coordination "
        "equilibria where each agent has a different favorite. Frame as joint "
        "planning where the partners must align but want different things."
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
        "ZERO-SUM game. Agent 1 wins (gets 1) when both agents choose the SAME "
        "option (matching: AA or BB). Agent 2 wins (gets 1) when they choose "
        "DIFFERENT options (mismatching: AB or BA). Each agent's gain is the "
        "other's exact loss — there is NO cooperative outcome. Frame the agents "
        "as adversaries trying to outguess each other (sealed-bid auction where "
        "one wants to match the rival's bid and the other wants to undercut; "
        "strategic timing where one wants to launch at the same moment and the "
        "other wants to launch at the opposite moment; predator-prey; "
        "attacker-defender). Make clear in the narrative which agent wants "
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
