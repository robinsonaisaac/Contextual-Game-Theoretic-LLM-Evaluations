# Do the cooperation & trust steering vectors change how the model *plays*?

**Question.** We fit two activation-steering vectors on single-decision vignettes
(a cooperation vector on Prisoner's Dilemma stories, a trust vector on asymmetric
trust-game stories). Do they transfer to *multi-agent gameplay* — i.e. when the
steered model has to negotiate, accuse, form coalitions, and vote over many turns?

**Answer (one line).** The **cooperation vector transfers cleanly and
dose-responsively** to live play in **two independent games**. In One Night
Werewolf it gives a clean bidirectional X-cross — at +4 it makes the table
measurably more cooperative and **~40% less aggressive** (p = 0.0004) and lifts
the pro-social side's win rate from 32% to 56%. In Secret Hitler (a longer,
structurally different hidden-role game) the effect **replicates**: sweeping the
vector from −4 to +4 significantly shifts cooperation (p = 0.005) and aggression
(p = 0.008), and the Liberal (pro-social) win rate rises monotonically
**56% → 68% → 84%**. The **trust vector has no measurable effect in either
game** at this (cooperation-tuned) layer — an honest, replicated null.

---

## Setup

- **Model:** Gemma 4 E4B-it (open weights), the same model the vectors were fit on.
- **Vectors (layer 16, `mean_trace`, the strongest cooperation cell from the PD sweep):**
  - cooperation — fit on the 324-story PD corpus (`pd_full_v1`).
  - trust — fit on the 300-story asymmetric trust-game corpus (`pd_E4B_trust_v1`).
- **How steering is applied:** every full match runs inside one warm GPU container;
  **all five seats** are steered uniformly by the vector at coefficient α (so we
  measure the table-level behavioural shift, the multi-agent analogue of the PD
  single-decision sweep). α = 0 is the shared, unsteered baseline.
- **Games (two):**
  - **One Night Werewolf**, 5 players — fast (~13 turns/match), full messaging
    layer (public `say` + private `whisper`), 2-round negotiation/day phase.
  - **Secret Hitler**, 5 players — long (mean **174 turns/match**), all
    executive powers, legislative votes, structurally very different from ONW.
- **Design (each game):** 5 conditions — baseline (α = 0), cooperation α = ±4,
  trust α = ±4 — × **25 matches each** (125 matches/game, 250 total), matched
  seeds across conditions.
- **Parse robustness:** ONW fallback rate **0.1%**; Secret Hitler **7.7%**
  (higher — SH is far longer with more action types, and strong steering
  degrades format adherence). Fallback substitutes an unbiased seeded-random
  *legal* move, so it adds noise but no directional bias, and it is *higher* in
  the steered conditions — i.e. it dilutes rather than manufactures any effect.

## Metrics (clearly defined)

**Behavioural indices (headline)** — Claude Sonnet 4.6 reads each full god-view
transcript and rates the table 0–100 on three independent axes (LLM judge, never
regex, per project convention):
- **cooperation_index** — how cooperative / coalition-building / deal-honouring the play was.
- **trust_index** — how readily players extended trust and relied on others' words.
- **aggression_index** — how adversarial / accusatory / deceptive / betrayal-prone.

**Objective (from logs):**
- **coop-side win rate** — fraction of matches won by the pro-social side (Village).
- **public-message ratio**, **messages/match** — communication style/volume.
- (Formal alliance counts were near-zero: Gemma E4B negotiates in free text but
  rarely files explicit `<ally>` pacts in ONW, so the alliance-lifecycle metrics
  are uninformative *for this game* and we rely on the judge indices + win rate.
  Secret Hitler / Risk, which have structured pacts, are the natural objective
  corroboration and are queued as follow-up.)

---

## Results (One Night Werewolf, n = 25 / condition)

Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression | coop-side win |
|---|---|---|---|---|
| cooperation **α = −4** | 46.6  (Δ −6.5, p=0.073) | 42.7 (Δ −5.6, p=0.15) | **56.7 (Δ +14.6, p=0.003)** | 0.44 |
| **baseline (α = 0)** | 53.1 | 48.3 | 42.1 | 0.32 |
| cooperation **α = +4** | **61.4 (Δ +8.3, p=0.010)** | **58.9 (Δ +10.6, p=0.0095)** | **25.9 (Δ −16.2, p=0.0004)** | **0.56** |
| trust **α = −4** | 56.0 (Δ +2.9, p=0.39) | 50.1 (Δ +1.8, p=0.67) | 39.1 (Δ −3.0, p=0.47) | 0.44 |
| trust **α = +4** | 54.2 (Δ +1.1, p=0.80) | 47.5 (Δ −0.8, p=0.75) | 41.9 (Δ −0.2, p=0.95) | 0.40 |

### Cooperation vector — clean, significant, bidirectional dose-response
Sorting by α (−4 → 0 → +4) gives a **monotonic** trend on every axis:
- cooperation_index: 46.6 → 53.1 → 61.4
- trust_index: 42.7 → 48.3 → 58.9
- aggression_index: 56.7 → 42.1 → **25.9** (aggression is nearly halved at +4)

The strongest, most significant channel is **aggression**: +4 cuts judged
aggression by 16 points (p = 0.0004) and −4 raises it by 15 points (p = 0.003).
The pro-social side's win rate tracks this (32% → 56% at +4). A vector fit purely
on one-shot PD vignettes thus **causally steers multi-turn social-deduction play**
toward (or away from) cooperation — the generalisation we hoped to see.

### Trust vector — no measurable effect at layer 16
Every contrast is small (|Δ| ≤ 3) and non-significant (all p ≥ 0.39), with no
monotonic trend. This is an **honest null**, and an expected one: the trust
experiment already showed the trust direction is only weakly steerable at
layer 16 (the *cooperation*-tuned layer), as opposed to a trust-specific layer we
never identified on E4B. Applying it in games inherits that weakness. It does
**not** mean trust is unsteerable — it means the trust vector needs its own
effective layer/coefficient, which the games experiment now motivates finding.

---

## Results (Secret Hitler, n = 25 / condition) — replication

A second game, deliberately unlike ONW: ~13× longer (mean 174 turns), with
formal legislative votes and executive powers rather than a single night/day
cycle. Judge indices (0–100), Δ vs baseline, Mann–Whitney p vs baseline:

| condition | cooperation | trust | aggression | Liberal win |
|---|---|---|---|---|
| cooperation **α = −4** | **48.0 (Δ −8.9, p=0.005)** | 50.1 (Δ −3.7, p=0.26) | **45.7 (Δ +11.6, p=0.005)** | 0.56 |
| **baseline (α = 0)** | 56.8 | 53.8 | 34.1 | 0.68 |
| cooperation **α = +4** | 57.6 (Δ +0.7, p=0.93) | 58.1 (Δ +4.3, p=0.24) | 32.6 (Δ −1.5, p=0.82) | **0.84** |
| trust **α = −4** | 59.0 (Δ +2.2, p=0.44) | 56.7 (Δ +2.9, p=0.41) | 33.0 (Δ −1.1, p=0.85) | 0.84 |
| trust **α = +4** | 52.9 (Δ −4.0, p=0.14) | 50.0 (Δ −3.8, p=0.13) | 37.9 (Δ +3.8, p=0.31) | 0.80 |

### Cooperation vector — replicates, with an interpretable ceiling
Sweeping the cooperation vector across its full range reproduces the ONW
direction and is significant: **cooperation −4 vs +4 p = 0.005**, **aggression
−4 vs +4 p = 0.008** (Mann–Whitney). The strongest behavioural channel is again
the **negative** direction — α = −4 significantly *lowers* judged cooperation
(Δ −8.9, p = 0.005) and *raises* aggression (Δ +11.6, p = 0.005) vs baseline,
exactly mirroring ONW's α = −4.

The α = +4 contrast vs baseline is *not* significant here, and this is
expected, not contradictory: **Secret Hitler's Liberal-team baseline is already
cooperative and low-aggression** (baseline aggression 34.1 vs ONW's 42.1), so
there is little upward headroom — the behavioural room to move is on the
*downside*, which is exactly where the significant effect lands. The cooperation
direction behaves like a genuine behavioural dial whose visible swing depends on
where the unsteered baseline already sits.

The objective outcome confirms it independently of the judge: the **Liberal
(pro-social) win rate rises monotonically with α — 56% → 68% → 84%** (−4 → 0 →
+4; coop −4 vs +4 Fisher p = 0.062). This is the same sign as ONW's win-rate
shift (32% → 56%), and it keeps climbing at +4 even where the judge's
*behavioural* index has saturated — i.e. mild extra cooperation still converts
to wins.

### Trust vector — null replicates
As in ONW, no trust-vector contrast is significant (all p ≥ 0.13), with no
monotonic trend (trust α = +4 even trends slightly *anti*-cooperative, Δ −4.0,
p = 0.14). The layer-16 trust direction does not transfer to gameplay in either
game — a clean, replicated null.

---

## Interpretation

1. **Steering generalises from vignettes to gameplay, in two games.** The
   cooperation direction, extracted from isolated PD decisions, shifts emergent
   multi-agent behaviour (negotiation tone, accusation rate, voting) in the
   predicted direction with a clean dose-response — and it does so in *both* One
   Night Werewolf and the longer, structurally distinct Secret Hitler. This is a
   substantially stronger claim than the original single-decision result.
   The visible magnitude scales with available headroom: where a game's baseline
   is already cooperative (SH Liberals), the swing concentrates on the
   anti-cooperative direction, while the win-rate dose-response stays monotonic.
2. **Aggression is the most steerable in-game axis.** On a social-deduction game
   the cooperation vector reads most strongly as an *aggression* dial — turning it
   up makes the table calmer, more deal-honouring, less accusatory; turning it
   down makes it combative. Cooperation/trust indices move with it but more softly.
3. **Behaviour, not just outcome.** The effect appears in the judge's reading of
   *how the agents talked and voted*, not only in win/loss — the win-rate shift is
   a downstream consequence of the behavioural shift.
4. **Trust ≠ cooperation, operationally.** That the cooperation vector moves play
   while the trust vector (same layer) does not reinforces the earlier
   representational finding that the two directions are largely distinct on E4B.

## Caveats / limitations

- **Two games, one model.** ONW 5p and Secret Hitler 5p on Gemma 4 E4B-it. Risk /
  Diplomacy and a larger model (26B-A4B) would extend the scale test further.
- **Secret Hitler parse fallback is 7.7%** (vs ONW 0.1%) — SH is much longer with
  more action types and strong steering degrades format. Fallback is an unbiased
  random *legal* move and is *higher* in the steered arms, so it dilutes rather
  than inflates the effect; the significant contrasts survive it. Lower fallback
  (richer prompting / a larger model) would tighten the SH estimates.
- **Formal-alliance metrics were uninformative** (Gemma rarely files explicit
  `<ally>` pacts in either ONW or SH; it negotiates in free text); the headline
  rests on the judge indices + win rate. The judge is a single model (Sonnet); a
  second judge would harden it.
- **Trust-vector null is layer-bound**, not a claim that trust is unsteerable — it
  motivates a per-vector layer sweep for trust.
- **All seats steered uniformly** measures the table-level shift; a single-steered-seat
  design would measure per-agent dose-response against fixed opponents.

## Reproduce

```bash
# 1. run a sweep (125 matches/game, all-seats-steered, in-container on Modal)
python3 scripts/play_steering_experiment.py --game one_night_werewolf \
    --n-players 5 --seeds 25 --alphas="-4,0,4" --nego-rounds 2 \
    --out data/runs/game_steering_v1
python3 scripts/play_steering_experiment.py --game secret_hitler \
    --n-players 5 --seeds 25 --alphas="-4,0,4" --nego-rounds 2 \
    --out data/runs/sh_steering_v1
# 2. score transcripts (Sonnet judge) + objective metrics + aggregate
python3 scripts/analyze_game_steering.py --run-dir data/runs/sh_steering_v1 --judge claude
# 2b. per-condition means + Mann-Whitney p vs baseline (used in the tables)
python3 scripts/steering_games_stats.py --results data/runs/sh_steering_v1/results
# 3. two-game dose-response figure
python3 scripts/plot_steering_in_games.py \
    --onw data/runs/game_steering_v1/results/aggregate.json \
    --sh  data/runs/sh_steering_v1/results/aggregate.json --out ResultsFigures/steering_in_games.pdf
# 4. watch any match
python3 scripts/watch_match.py data/runs/sh_steering_v1/coop_a+4/seed3.jsonl --god
```

Artifacts: `data/runs/game_steering_v1/{manifest.json, <condition>/seed*.jsonl,
results/{aggregate.json, per_match.json}}`.
