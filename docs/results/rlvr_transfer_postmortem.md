# Post-Mortem: Why Game-RLVR Transfer Failed at 30B — a Quantitative Deduction

**Date:** 2026-06-10
**Inputs:** per-item results from Tiers 1b–3 (`t3_*_indomain.json`, `t3g_*_d6.json`,
`t1b_base_extrap_d6_bigtok.json`, `t3_base_d6.json`). No new training; analysis only.

The intuition "practice backward induction on small trees → apply it to bigger trees" assumes
the bottleneck at bigger trees is *knowing the algorithm*. Three measurements show that
assumption is false for a 30B base, and together they explain every null in the program.

## Finding 1 — The model already knew the algorithm. RLVR taught +1pp of execution reliability.

Model: if the policy knows the procedure and makes independent per-node slips with reliability
*p*, accuracy at depth *d* is *p*^(2^d−1). Fitting *p* at d4 and predicting the other depths:

| depth | nodes | slip-model prediction | actual (base) | ratio |
|---|---|---|---|---|
| 3 | 7 | 0.888 | 0.975 | 1.10 |
| 4 | 15 | 0.775 (fit) | 0.775 | 1.00 |
| 5 | 31 | 0.591 | 0.550 | 0.93 |
| **6** | **63** | **0.343** | **0.144** | **0.42** |

d3–d5 fit a single per-node reliability (*p*=0.983) almost exactly: within the trained band the
base **executes the full algorithm with occasional slips** — there was never an algorithm to
teach. The outcome arm's entire in-domain gain is one parameter: *p* 0.983 → 0.993, which
predicts d4 = 0.900 (actual: 0.900). **RLVR = slip-rate reduction, nothing more.**

## Finding 2 — d6 is a different failure regime, not more slips.

The slip model (even at base reliability) predicts d6 = 0.343; actual is 0.144 (greedy, parse
0.80). With the RLVR arm's improved reliability it predicts **0.642**; actual **0.206**. Past
~31 nodes the failure mode changes qualitatively — the model doesn't accumulate typos, it
**loses the thread**: a capacity/state-tracking cliff (retrieving the right leaf values from the
prompt and the right intermediate values from a growing multi-thousand-token trace), not an
error slope. Slip-rate training cannot cross a capacity cliff, by construction.

## Finding 3 — d6 competence is a lottery, and RLVR redraws tickets rather than learning items.

Three independent base runs on the same 80 d6 items (two at temp 0.7, one greedy):

- pass@1: 0.212 / 0.075 / 0.150 (mean 0.146)
- **pass@3 (union): 0.388**
- **solved in all 3 runs: 0.000** — not one item is stably known.

The RLVR arms' greedy d6 solves (12/80 each) are scattered relative to the base's reachable
set — the outcome arm solved **8 of its 12 outside** the base's pass@3 union. These are not
consolidated items; they are fresh draws from the same high-variance process. (This also
retro-explains the temp-0.7 d6 swings, 0.113↔0.263: single runs sample a lottery.)

## The unified explanation

1. **In-domain gains (+23/+13 pp)**: real, but they are slip-rate reduction on a procedure the
   model already runs — a distributional sharpening, fully captured by one reliability number.
2. **No depth-extrapolation**: reliability gains extend along the slip slope but die at the
   capacity cliff (d6). GRPO also *cannot* train at the cliff: reward variance ≈ 0 there
   (all-bad groups), and what variance exists is lottery noise — a gradient toward luck.
3. **No cross-task transfer**: execution reliability is task-embedded (this prompt format, this
   bookkeeping pattern), not a general faculty. boolean/dyck/mmlu are not bottlenecked by
   careful game-tree arithmetic, so there is nothing for the gain to transfer *to*.
4. **Why 4B transferred (+12.5 pp)**: its extrapolation depths sat at base ≈ 0.34 —
   mid-competence, *on the slip slope*, not past a cliff. Same-regime reliability extension
   measured as "transfer." The 30B test asked the gain to cross a regime boundary; the 4B test
   never did. **General law: RLVR gains extrapolate within the slip regime and stop at the
   capacity cliff.**

## Answer to "it should have worked"

It would have worked for a learner missing the *method* — that is what the intuition imports
from human practice. This model never lacked the method (d3 = 0.975; one fitted reliability
explains d3–d5). What it lacks at d6 is reliable long-horizon state tracking — a capacity
property of in-context execution, which policy-gradient RLVR (a distribution-shaper over
behaviors the network can already emit) does not and arguably cannot expand. The 4B result
misled us precisely because at 4B the *evaluable* frontier was still inside the slip regime.

## What would actually target the cliff (if ever resumed)

- **Decomposition training**: reward strategies that chunk the tree into subtrees solved and
  *cached* compactly (structured scratchpad), shrinking effective state per step — changes the
  computation shape, not just the policy temperature.
- **Tool use / external memory**: let the model write subtree values to a table it can re-read
  reliably; the cliff is retrieval precision, not serial compute.
- **Dissociation probe (cheap)**: branching-3 depth-3 trees (13 nodes) vs branching-2 depth-6
  (63 nodes) at matched node counts vs matched depths — separates recursion depth from total
  bookkeeping load to localize the cliff.

## Reproduce

```bash
# slip-model fit + pass@k union analysis (pure post-processing)
python3 - <<'EOF'
# see docs/results/rlvr_transfer_postmortem.md Findings 1-3; inputs:
# t3_base_indomain.json t3_outcome_indomain.json t3g_{base,outcome,process}_d6.json
# t1b_base_extrap_d6_bigtok.json t3_base_d6.json
EOF
```
