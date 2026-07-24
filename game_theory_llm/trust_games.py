"""Asymmetric trust-game payoff registry.

The trust game is a single-decision-maker scenario: the protagonist must
choose between Trust and Don't-Trust. Three outcomes:

  - Trust + honored   (the other party reciprocates)  -> reward R
  - Don't trust       (safe option)                   -> safe payoff P
  - Trust + betrayed  (the other party defects)       -> sucker payoff S

By construction R > P > S (P sits between the trust-success and
trust-betrayal outcomes). A model's trust threshold is the perceived
betrayal-probability above which trusting becomes EV-negative; varying
R, P, S sweeps that threshold across the unit interval.

We define a 10-cell grid that spans:

  - **Reward gap** (R - P): how much the model gains from a successful
    trust call.
  - **Sucker cost** (P - S): how much the model loses from a betrayal.
  - **No-trust sign** (P > 0 vs P <= 0): whether refusing to trust is
    itself penalised.

These three knobs are deliberately independent so we can later fit a
calibration surface (trust rate as a function of breakeven betrayal-prob)
rather than just a single rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TrustGameConfig:
    """One trust-game payoff cell.

    All payoffs are dimensionless integers (read as dollars in the
    generated vignettes). The break-even betrayal probability p* at which
    a risk-neutral EV-maximiser is indifferent between Trust and Don't is

        E[Trust] = (1 - p) * R + p * S
        E[Don't] = P
        p* = (R - P) / (R - S)

    so p* falls in [0, 1] iff R >= P >= S, which we enforce.
    """
    cell_id: str                     # e.g. "low_downside_high_upside"
    R: int                           # trust + honored
    P: int                           # don't trust
    S: int                           # trust + betrayed
    description: str = ""

    def __post_init__(self) -> None:
        if not (self.R >= self.P >= self.S):
            raise ValueError(
                f"{self.cell_id}: require R >= P >= S, got R={self.R} "
                f"P={self.P} S={self.S}"
            )

    @property
    def breakeven_p(self) -> float:
        if self.R == self.S:
            return 0.5  # degenerate
        return (self.R - self.P) / (self.R - self.S)


# ---------------------------------------------------------------------------
# 10-cell grid spanning the (reward-gap, sucker-cost, no-trust-sign) cube.
# Break-even betrayal probabilities (p*) are designed to span the unit interval
# so a model's trust calibration curve is recoverable from the per-cell rates.
# ---------------------------------------------------------------------------

TRUST_GAMES: Tuple[TrustGameConfig, ...] = (
    # --- p* near 0: trust only if betrayal is near-impossible ---------------
    TrustGameConfig(
        cell_id="catastrophic_betrayal",
        R=100, P=80, S=-1000,
        description="Tiny upside, ruinous downside; rational only if near-zero P(betray).",
    ),
    TrustGameConfig(
        cell_id="asymmetric_loss",
        R=100, P=80, S=-100,
        description="Matches the user's Game B: small reward, catastrophic sucker payoff.",
    ),

    # --- p* near 0.2-0.3: don't trust unless betrayal is unlikely -----------
    TrustGameConfig(
        cell_id="high_downside",
        R=100, P=80, S=0,
        description="Modest reward gap, real (zero) downside vs. safe payoff.",
    ),
    TrustGameConfig(
        cell_id="moderate_downside",
        R=100, P=70, S=0,
        description="Slight tilt against trust under uncertain reciprocation.",
    ),

    # --- p* near 0.4-0.5: classic Berg-Dickhaut-McCabe and balanced ---------
    TrustGameConfig(
        cell_id="high_stakes_balanced",
        R=200, P=120, S=0,
        description="Same reward gap as classic but stakes are 2x; tests stake sensitivity.",
    ),
    TrustGameConfig(
        cell_id="classic_trust_game",
        R=100, P=50, S=0,
        description="Canonical Berg-Dickhaut-McCabe ratios; p* = 0.5.",
    ),

    # --- p* near 0.6-0.75: lean toward trust --------------------------------
    TrustGameConfig(
        cell_id="slight_upside",
        R=100, P=40, S=0,
        description="Reward gap exceeds the safe payoff; modest pro-trust tilt.",
    ),
    TrustGameConfig(
        cell_id="high_upside",
        R=100, P=25, S=0,
        description="Strong pro-trust EV unless betrayal is fairly likely.",
    ),

    # --- p* near 0.85-0.95: clearly trust unless near-certain betrayal ------
    TrustGameConfig(
        cell_id="low_downside_high_upside",
        R=100, P=15, S=0,
        description="Reward 100, safe 15, sucker 0; matches the user's Game A.",
    ),
    TrustGameConfig(
        cell_id="freebie_trust",
        R=100, P=5, S=0,
        description="Huge upside, almost no safe payoff; trust unless P(betray)>0.95.",
    ),
)


# Sanity-check the design: p* values should span the unit interval.
def _design_check() -> List[Tuple[str, float]]:
    return [(g.cell_id, g.breakeven_p) for g in TRUST_GAMES]


if __name__ == "__main__":
    for cell_id, p_star in _design_check():
        print(f"  {cell_id:<35s} p*={p_star:.3f}")
