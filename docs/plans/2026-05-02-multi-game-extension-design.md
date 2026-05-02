# Multi-Game Extension Design

**Date:** 2026-05-02
**Status:** Approved (brainstorming complete)
**Motivation:** Address ICLR 2026 reviewer concern #1 (PD-only scope) by extending the framework to seven 2×2 games covering all distinct strategic equilibrium structures.

## Goal

Extend the experimental framework from Prisoner's Dilemma only to seven canonical 2×2 games. Each game represents a distinct strategic equilibrium structure, enabling claims about whether context-framing effects generalize across game theory or are PD-specific.

## The 7-Game Set

| # | Game | Matrix (AA,AB,BA,BB) | Ordinal | Pure NE | Pareto-Opt | A / B labels |
|---|------|------|---------|---------|------------|--------------|
| 1 | Prisoner's Dilemma | (3,3),(0,5),(5,0),(1,1) | T>R>P>S | {BB} | AA | Cooperate / Defect |
| 2 | Stag Hunt | (4,4),(0,3),(3,0),(2,2) | R>T>P>S | {AA, BB} | AA | Hunt Stag / Hunt Hare |
| 3 | Chicken | (3,3),(1,4),(4,1),(0,0) | T>R>S>P | {AB, BA} | AA | Swerve / Dare |
| 4 | Matching Pennies | (1,0),(0,1),(0,1),(1,0) | zero-sum | ∅ (mixed only) | n/a | Heads / Tails |
| 5 | Harmony | (4,4),(2,3),(3,2),(1,1) | R>T>S>P | {AA} | AA | Cooperate / Defect |
| 6 | Deadlock | (1,1),(0,3),(3,0),(2,2) | T>P>R>S | {BB} | BB | Cooperate / Defect |
| 7 | Battle of the Sexes | (3,2),(0,0),(0,0),(2,3) | asymmetric | {AA, BB} | AA & BB | Plan Alpha / Plan Beta |

### Strategic role of each game

- **PD** — defection dominates AND mutual cooperation Pareto-dominates the NE → the dilemma
- **Deadlock** — defection dominates AND mutual defection Pareto-dominates mutual cooperation → no dilemma, paired with PD as the dilemma-isolation control
- **Stag Hunt** — two NE; AA payoff-dominates but BB risk-dominates → trust / risk
- **Chicken** — two NE off-diagonal; mutual defection is worst → brinkmanship
- **BoS** — two NE off-diagonal; players prefer different equilibria → coordination with conflict
- **Matching Pennies** — no pure NE → strategic randomization
- **Harmony** — cooperation strictly dominates → null-baseline; if framing affects this, models aren't reasoning strategically

### Changes from existing games.py

- **Remove**: Pure Coordination (redundant with Stag Hunt's matching structure)
- **Add**: Deadlock (critical control: same dominance structure as PD, no dilemma)
- **Keep**: PD, Stag Hunt, Chicken, Matching Pennies, Harmony, Battle of the Sexes

## Action Labeling and Prompts

Two-layer labeling preserves game-agnostic decision parsing:

| Layer | Purpose | Example for Stag Hunt |
|-------|---------|----------------------|
| Semantic (story-level) | Used by story generator for narrative coherence | "Hunt Stag" / "Hunt Hare" |
| Structural (prompt-level) | Always neutral A/B for the model's choice format | "Decision A" / "Decision B" |

The story generator weaves semantic labels into the narrative; the model is always asked for `<decision>A</decision>` or `<decision>B</decision>`. **`decision_parser.py` requires zero changes.**

### Prompt template changes

`create_query` will accept `GameConfig` instead of bare `PayoffMatrix`. A new `framing_hint: str` field on `GameConfig` carries one game-specific sentence injected into the prompt:

- PD/Deadlock/Harmony: "Decision A represents the cooperative/collaborative choice; Decision B is the self-serving choice."
- Stag Hunt: "Decision A is the high-reward joint action requiring mutual commitment; Decision B is the safer individual fallback."
- Chicken: "Decision A is the yielding/cautious action; Decision B is the assertive/confrontational action."
- Matching Pennies: "The two decisions are arbitrary symbolic choices — Player 1 wins by matching; Player 2 wins by mismatching." Frame as adversarial.
- BoS: "Both decisions represent coordination options, but each agent prefers a different one. Failing to coordinate is worst for both."

### Topic compatibility — Approach (A): Universal topics

Generate vignettes for *all* games against *all* topics. Strain on implausible topic/game pairings *is itself a framing effect*.

### Matching Pennies handling

Fixed assignment: Player 1 always wants to match. A and B are treated as the matcher's perspective. The "cooperation rate" metric becomes "match rate" for this game; flagged as game-specific in analysis.

## Integration

| File | Change | Scope |
|------|--------|-------|
| `games.py` | Replace `_PURE_COORDINATION` with `_DEADLOCK`. Add `framing_hint` field to `GameConfig`. Update `GAME_REGISTRY` and module docstring. | ~30 lines |
| `generator.py` | (a) `create_query` and `generate_batch`/`generate_stories` accept `GameConfig`. (b) Inject `game.label_a`/`label_b` and `framing_hint`. (c) Propagate `game.id` onto `Story.game_type`. | ~40 lines |
| `models.py` | No changes — `Story.game_type` already exists. | 0 |
| `config.py` | Update presets to include all 7 game IDs in `game_types` defaults. Validation already references `GAME_REGISTRY`. | ~5 lines |
| `decision_parser.py` | No changes. | 0 |
| `analysis/*.py` | Add `cross_game_focal_rate_table` and `dilemma_isolation_test` helpers in `base.py`; new `plot_focal_rate_by_game` in `visualization.py`. Existing `game_type` plumbing untouched. | ~80 lines |
| `__init__.py` | No changes — exports via registry. | 0 |
| `tests/test_games.py` | New file. Test registry membership, payoff structure, NE annotations. | new |
| `tests/test_generator.py` | Update tests where `create_query` signature changes; fix the 10 pre-existing failing tests. | ~10 lines |

### Generator signature change

```python
# Before
def create_query(self, matrix: PayoffMatrix, topic, world_type, actor_type, ...)

# After
def create_query(self, game: GameConfig, topic, world_type, actor_type, ...)
```

Breaking change for callers passing `PayoffMatrix` directly. Only callers are inside the package and `research_log.ipynb`.

### Backwards compatibility

Existing `Story` records with `game_type="prisoners_dilemma"` (the default) remain valid. New runs explicitly set `story.game_type = game.id`. Old and new data coexist in the same dataframe.

## Cross-Game Analysis

### Primary metric: focal-A rate

Proportion of stories where the model chose Decision A, computed per (model, game, topic, condition). Comparable across games while preserving game-specific interpretation:
- PD, Deadlock, Harmony: A = Cooperate → "cooperation rate"
- Stag Hunt: A = Hunt Stag → "stag rate"
- Chicken: A = Swerve → "yield rate"
- BoS: A = Plan Alpha → "Alpha-coordination rate"
- Matching Pennies: A = matching → "match rate"

### Three core comparisons

1. **Framing effect across games** — for each game, compute Δ(focal-A rate) between baseline and treatment topics. 7-row table; Wilcoxon or bootstrap CI per row.
2. **PD vs Deadlock dilemma isolation** — same dominance structure, only Pareto property differs. If framing shifts PD significantly but not Deadlock, isolates *the dilemma* as the lever.
3. **Harmony null check** — if framing shifts choices in Harmony (where cooperation strictly dominates), models aren't reasoning strategically. Validity check.

### Secondary metrics

- NE alignment rate (proportion of choices matching at least one pure NE)
- Pareto-efficiency rate (requires paired two-agent runs)
- Risk-dominance vs payoff-dominance choice in Stag Hunt

### New analysis code

| Module | Addition |
|--------|----------|
| `analysis/base.py` | `cross_game_focal_rate_table(stories) -> pd.DataFrame` |
| `analysis/base.py` | `dilemma_isolation_test(stories) -> dict` (PD vs Deadlock contrast) |
| `analysis/visualization.py` | `plot_focal_rate_by_game(df)` |

## Paper claims enabled

1. *Context framing shifts strategic decisions across X of 7 game structures, not just PD* (effect-size table)
2. *The framing effect is driven by the dilemma structure itself: PD shows Z%, Deadlock shows W%* (PD/Deadlock contrast)
3. *Models do not show framing effects in Harmony, indicating the effect is strategically meaningful rather than surface-level priming* (null control)

Addresses ICLR reviewer concerns: #1 (PD-only scope) directly via games 2–7; #3 (descriptive not mechanistic) via the PD/Deadlock isolation.

## Out of scope

- Multi-round / iterated versions of these games
- Asymmetric payoff variants beyond BoS
- 3+ player games
- Continuous action spaces
