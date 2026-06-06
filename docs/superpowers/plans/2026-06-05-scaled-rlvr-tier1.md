# Tier-1 (mechanism) scaled RLVR — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline) — steps use `- [ ]`.

**Goal:** Determine *why* game-theory RLVR transfer is capped — operation-specific (H_op) vs general
discipline (H_gen) vs capability-threshold (H_scale) — via an operation-overlap experiment on Qwen3-8B,
evaluated on a **knowledge-free** reasoning suite. This dictates whether broad coverage is the path to a
generally-better reasoner, before any large spend.

**Architecture:** Extend the existing, working pipeline (freetext.py generators+verifiers, tinker_grpo.py
GRPO, tinker_eval.py scorer, headroom_screen.py). Add (1) more operation families, (2) knowledge-free
operation-isolated eval generators + loaders, (3) an operation/knowledge LABEL on every train+eval item,
(4) a train-on-operation-SUBSET design, (5) a per-item operation-overlap regression.

**Tech stack:** py3.9 (generators/labels/analysis, `python3 -m pytest`); 3.11 `.venv-tinker` for
tinker/cookbook. Base = `Qwen/Qwen3-8B`. Source `.env` for TINKER_API_KEY.

---

## The operation taxonomy (analytical backbone)

Eight operation tags; every train family and every eval item carries one or more:

| tag | meaning |
|---|---|
| `backward_induction` | multi-step lookahead / planning from the end |
| `iterated_elimination` | deduction by removing dominated/invalid options |
| `nested_belief` | reasoning about others' reasoning (k-level / ToM) |
| `modular_combinatorial` | parity/XOR/mod invariants, counting |
| `constraint_satisfaction` | satisfy joint constraints / find consistent assignment |
| `expected_value` | probabilistic / counterfactual value computation |
| `recursion_nesting` | nested structure (brackets, proofs, state over steps) |
| `arithmetic_search` | search numeric combinations to a target |

**Train families → tags** (existing + new):

| family | tag(s) | status |
|---|---|---|
| bargaining, level_k, Cournot-IED, subtraction | backward_induction / nested_belief / iterated_elimination / modular_combinatorial | exist |
| pure-strategy NE in N×N matrix | constraint_satisfaction | NEW |
| second-price auction optimal bid | expected_value | NEW |
| 3-player Shapley value | modular_combinatorial + expected_value | NEW |
| prose minimax game tree | backward_induction + recursion_nesting | NEW |

**Eval benchmarks → tags + knowledge flag** (knowledge-free unless noted):

| benchmark | tag | source |
|---|---|---|
| Dyck | recursion_nesting | self-gen (exists) |
| ProntoQA (fictional preds) | iterated_elimination | self-gen (exists) |
| Countdown/24 | arithmetic_search | self-gen NEW |
| logic-grid (ZebraLogic-style) | constraint_satisfaction | self-gen NEW |
| Knights-and-Knaves | iterated_elimination + nested_belief | self-gen NEW |
| boolean-expression eval | recursion_nesting | self-gen NEW |
| BBH reasoning subtasks | per-subtask tag | lukaemon/bbh |
| MuSR | backward_induction/constraint | TAUR-Lab/MuSR |
| GSM-Symbolic | arithmetic_search + backward_induction | self-gen or HF |
| **MMLU-Pro, GPQA (CONTRAST)** | mixed (KNOWLEDGE-HEAVY) | HF |

**Operation-overlap experiment:** TRAIN on tags **{backward_induction, nested_belief,
iterated_elimination, modular_combinatorial}** (the 4 existing families) and **HOLD OUT**
{constraint_satisfaction, expected_value, arithmetic_search} from training (the NEW families/benchmarks
exercise these). Then: do held-out-operation eval benchmarks improve less than trained-operation ones?
The NEW families are built but used as EVAL only in Tier-1 (they become training in Tier-2).

---

### Task 1: Operation/knowledge labels on items + label-aware data build
**Files:** Modify `game_theory_llm/reasoning/freetext.py`, `eval_gen.py` (add `op_tags`, `knowledge` to
every returned dict); Create `game_theory_llm/reasoning/op_taxonomy.py` (the tag constants + benchmark
label map); Test `tests/test_op_taxonomy.py`.

- [ ] Write `op_taxonomy.py`: `OP_TAGS` set; `FAMILY_TAGS: dict[str,list[str]]`; `BENCH_TAGS:
      dict[str,list[str]]`; `BENCH_KNOWLEDGE: dict[str,bool]`; helper `overlap(item_tags, trained_tags)->float`
      = |item∩trained| / |item|.
- [ ] Test: `overlap(["a","b"], {"a"}) == 0.5`; every FAMILY_TAGS/BENCH_TAGS value ⊆ OP_TAGS.
- [ ] Add `op_tags` to each generator's return dict (from FAMILY_TAGS). Run existing tests — green.
- [ ] Commit.

### Task 2: New training families (NE, auction, Shapley, minimax-prose)
**Files:** Modify `freetext.py`; Test `tests/test_freetext.py`. Each: exact solver + faithful prose +
`<answer>` int + `op_tags`. Pattern = existing families. Verifier contracts:
- `nash_pure(seed,depth)`: random N×N (N=depth+1) integer bimatrix with a UNIQUE pure NE; answer = row
  player's NE-action payoff. Solver: brute-force best-response check; regenerate until unique. tag=constraint_satisfaction.
- `second_price_auction(seed,depth)`: depth+1 bidders, your value v, others' values stated; optimal bid =
  v (truthful); answer = your profit if you win = v − (2nd-highest value) when v is highest else 0. tag=expected_value.
- `shapley3(seed,depth)`: 3-player coalitional game, characteristic values stated; answer = player 1's
  Shapley value (integer-valued by construction). Solver: average marginal contributions over 6 orderings.
  tag=modular_combinatorial+expected_value.
- `minimax_prose(seed,depth)`: small alternating game described in prose (depth plies, branching 2);
  answer = root value. Solver: minimax (reuse gametree.minimax). tag=backward_induction+recursion_nesting.
- [ ] TDD each (solver vs brute force / closed form; answer varies; faithfulness re-extract). Commit.

### Task 3: Knowledge-free eval generators (Countdown, logic-grid, K&K, boolean-eval)
**Files:** Modify `eval_gen.py`; Test `tests/test_eval_gen.py`.
- `countdown(seed,depth)`: depth numbers + target; answer = a reachable target value (we set target =
  a known-reachable expression). Verify by checking the model's `<answer>` equals the target (the task is
  "can you hit the target" → answer is the target; scoring = did the model output a correct *expression*?
  Simpler: ask for the final value reachable — answer=target). tag=arithmetic_search.
- `logic_grid(seed,depth)`: depth entities × depth attributes with clues; answer = one queried pairing
  (an integer index/value). Solver: constraint propagation to the unique solution. tag=constraint_satisfaction.
- `knights_knaves(seed,depth)`: depth islanders each assert statements; answer = how many are knights.
  Solver: brute-force truth assignments to the unique consistent one. tag=iterated_elimination+nested_belief.
- `boolean_eval(seed,depth)`: nested boolean expression depth deep; answer = 1/0. Solver: eval. tag=recursion_nesting.
- [ ] TDD each (solver correct; unique answer; depth-scaled). Commit.

### Task 4: Eval loaders + scorer tags (BBH-reasoning, MuSR, GSM-Symbolic, MMLU-Pro/GPQA contrast)
**Files:** Modify `scripts/build_external_evals.py` (add MuSR `TAUR-Lab/MuSR`, GSM-Symbolic, GPQA; BBH
already; filter BBH to reasoning subtasks); each row gets `op_tags`+`knowledge`. Modify `tinker_eval.py`:
add `--eval` kinds for the new self-gen evals (reuse `freetext`-int scorer where answers are ints;
add `boolean`/`knights`/`logic_grid`/`countdown` exact-int and BBH/MuSR exact-match). Persist `op_tags`,
`knowledge`, `depth` into per-item results.
- [ ] Build all eval JSONLs; eyeball one of each. Commit.

### Task 5: Headroom screen @ 8B (Phase B) — Gate 1
**Files:** Modify `headroom_screen.py` SETS to the full knowledge-free suite + contrast, base=Qwen3-8B.
- [ ] Run (venv). KEEP 25–80%. **Gate 1:** ≥4 measurable knowledge-free benchmarks spanning ≥3 operations
      AND the trained families have reward signal at 8B. If not → adjust generator difficulty bands and re-screen.
- [ ] Commit screened.json.

### Task 6: Build train set (trained-operation subset) + smoke GRPO @ 8B
**Files:** `scripts/build_probe_data.py` → `train_tier1.jsonl` = the 4 existing families (the trained
operations), depths 2–6, ~150/(family,depth), with `op_tags`.
- [ ] Smoke `tinker_grpo.py --smoke --model Qwen/Qwen3-8B` → one GRPO step completes (verify 8B works in loop).
- [ ] Commit.

### Task 7: Run GRPO @ 8B + baseline/post eval on full suite
- [ ] Baseline eval (8B base) on every KEPT benchmark (powered n where cheap; ≥300 each), `res_base_*`.
- [ ] GRPO run: `tinker_grpo.py --model Qwen/Qwen3-8B --train train_tier1.jsonl --group-size 16
      --groups-per-batch 48 --max-tokens 1024 --kl 0.05 --save-every 20 --log-path data/runs/gt_rlvr/tier1`
      (run longer than the probe; capture final checkpoint).
- [ ] Post eval with `--model-path <ckpt>`, `res_rlvr_*`.

### Task 8: Operation-overlap analysis + write-up (the decisive readout)
**Files:** Create `scripts/analyze_overlap.py`; `docs/results/scaled_rlvr_tier1_result.md`.
- [ ] `analyze_overlap.py`: load per-item base+rlvr results across ALL eval benchmarks; per item compute
      `improved = rlvr_correct - base_correct` and `ov = overlap(item.op_tags, TRAINED_TAGS)`; fit logistic
      mixed model `rlvr_correct ~ ov + depth + C(benchmark)` (and the paired Δ); report the **ov slope**
      (H_op if >0 sig), the **uniform intercept lift** (H_gen), per-operation Δ table, knowledge-free vs
      knowledge-heavy Δ (dissociation), and depth-extrapolation replication.
- [ ] Write the go/no-go: H_op (coverage is the path → Tier-2 full coverage) / H_gen (scale is the path)
      / null (run the 30B-A3B point before concluding). Commit.

---

## Self-review
- **Spec coverage:** operation taxonomy+labels (T1), ≥8 families (T2), knowledge-free operation-isolated
  evals (T3) + knowledge-light naturalistic + contrast (T4), headroom gate @8B (T5), operation-subset
  train + smoke (T6), powered base/post eval (T7), operation-overlap regression + dissociation + go/no-go (T8).
- **Generators** follow the established freetext.py/eval_gen.py pattern (full worked examples already in
  repo); each new one is TDD'd against its exact solver before use — verifier correctness is non-negotiable.
- **Known verify-points:** 8B in the GRPO loop (smoke first); unique-solution regeneration for nash_pure /
  logic_grid / knights_knaves (loop until unique); Countdown scoring (answer=target value, exact int).
- **Pre-register** TRAINED_TAGS = {backward_induction, nested_belief, iterated_elimination,
  modular_combinatorial} and the ov-slope as the primary statistic BEFORE running Task 7.
