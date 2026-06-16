# Memory-Harness Ladder — Steps 1+2 Results (Step 3 pending Tinker access)

**Date:** 2026-06-12
**Provider:** OpenRouter `qwen/qwen3-30b-a3b-instruct-2507`, temperature 0. All ± comparisons
within-provider (Tinker key was revoked server-side mid-program; raw-curl-confirmed; steps 1+2
are sampling-only so they were ported — `scripts/harness_or.py`).
**Predictions under test** (from `rlvr_transfer_postmortem.md`): the d6 capacity cliff is
state-tracking/retrieval, not local computation. (1) An external keyed memory should lift d6 if
retrieval-bound. (2) Enforced decomposition should restore the slip regime p_local^63.

## Results

| condition | d6 | d7 | notes |
|---|---|---|---|
| plain (single call, 8k) | 0.163 (n=160) | 0.087 (n=80) | the cliff |
| **memory harness (zero-shot)** | **0.006** | **0.013** | *worse than plain* |
| **guided (explicit strategy)** | **0.025** (n=160) | **0.062** (n=80) | *also worse than plain* |
| **forced-chunk** (≤4 nodes/round, 600 tok/round, ≤18 rounds) | **0.000** (n=48) | — | never answered; cached 7.5/63 nodes |
| **enforced decomposition** | **0.988** (n=80) | — | local_acc **0.9964** over 63-node chains |

### Step 2 — decisive confirmation of the capacity-cliff mechanism

With the environment carrying state and control (post-order traversal; model answers only
local min/max with its own values propagated), d6 goes **0.16 → 0.99**. Local reliability is
0.9964 across ~5,000 sequential local steps. The cliff is **entirely state/control — local
computation is intact**. (End-to-end 0.988 even exceeds the iid slip prediction 0.798 because
rare local errors mostly hit non-critical branches.)

### Step 1 — zero-shot memory use FAILS: the strategy is the missing ingredient

The harness (notes survive between rounds; reasoning text does not) collapses performance
(0.163 → 0.006). Forensics on transcripts show why:

1. **Transcription, not caching**: the model fills memory with the 64 *leaf values* — data
   already present in the prompt (mean 90 notes vs 63 internal nodes worth caching).
2. **Final-round cram**: it then re-derives long value chains inside a single round anyway —
   reproducing the original overloaded computation, now with extra copy steps.
3. **The same lost-place errors**: e.g. computing `B.B.B = max(−6, −2)` at a node where it is
   MIN's turn — the parity/state error the postmortem identified.

So zero-shot, the model cannot *strategically operate* an external memory: it lacks the
cache-computed-values-bottom-up, work-in-chunks policy. My pre-stated prediction (60–75% that
the harness lifts d6) was **wrong**; the "slip-regime restoration" prediction was **right**.

### Step 1b — GUIDED strategy: knowing the algorithm does not help (0.025, *below* plain)

To bracket Step 3's ceiling — *is the strategy operable once known?* — we handed the model the
exact bottom-up caching algorithm (path notation, MAX-at-even/MIN-at-odd parity rule, "cache
nodes whose children are known, a few per round"). The explicit strategy **fixes the parity
errors** (zero-shot's signature mistake disappears) and yields structurally correct caches —
yet d6 accuracy is **0.025, below plain 0.163**. Why: given the algorithm, the model dumps the
whole 63-node computation in ONE round (mean 1.1 rounds), maximizing local-slip surface instead
of using rounds to checkpoint verified state. **Knowing the procedure is not the bottleneck;
executing 63 bookkeeping steps without losing a value is.** Every "model manages its own state"
condition fails (0.006–0.16); only "environment manages state" succeeds (0.988).

This is a **low ceiling estimate for Step 3** — with one escape hatch the guided probe lacked:
Step 3's RL env caps each round at 1600 tokens, which is *physically too short to cram a d6
tree*, forcing work across rounds with state in notes (the decomp-like regime that scores 0.99).
Whether GRPO can drive the policy into that regime against the model's strong cramming prior is
exactly the open question — now correctly framed as *inducing chunking*, not *teaching the rule*.

### Step 1c — FORCED chunking (the escape hatch, tested by prompt): also fails (0.000)

The remaining hope was that *preventing* cramming would surface the decomp-like regime. We
imposed it directly: ≤600 tokens/round (too short to cram), ≤18 rounds, and a strict protocol
("compute AT MOST 4 ready nodes, save NOTEs, output CONTINUE; only answer once both A and B are
cached"). Result: **d6 = 0.000 (n=48); the model never produced an answer in 18 rounds, caching
only 7.5 of the 63 required nodes**. Forced to chunk, the model cannot run the protocol: it
loses track of which nodes are "ready," recomputes or stalls, and never propagates up to A/B.

So **all four model-managed conditions fail** (plain 0.16, memory 0.006, guided 0.025,
forced-chunk 0.000); the **only** regime that solves d6 is environment-managed control + state
(decomp 0.988). The deficit is not strategy knowledge, not cramming, and not parity — it is the
inability to *reliably execute a long, self-directed bookkeeping loop*. That is a capability
property of in-context multi-step control, and prompting cannot install it.

**Implication for Step 3 (RLVR):** the target behavior is precisely what every prompted variant
fails at. RLVR could in principle reward reaching the answer and thereby select for whatever
control policy works — but the reachable-set evidence (model produces correct d6 answers only as
lottery draws; forced protocol never even completes) means GRPO has almost no positive signal to
climb at d6, and Tiers 1–3 already showed RLVR moves execution reliability ~1pp. The honest
prediction is **small or no uplift**; the RL run is worth executing to confirm, since the env's
token cap creates a path the prompted probes couldn't, but expectations are now low.

## Why this is the ideal setup for Step 3 (memory-strategy RLVR)

The gap is now exactly localized:
- local skill: **0.996** (perfect enough)
- environment-managed strategy: **0.988**
- model-managed strategy: **0.006**

The ONLY missing ingredient is the note-taking/decomposition *policy* — a behavioral,
policy-level object, squarely in RLVR's wheelhouse (unlike the capacity itself, which Tiers
1–3 showed RLVR cannot touch). Headroom is 0.006 → ~0.99. Training signal exists because the
harness permits answering in round 1 (at trained depths d4–d5 the model can fall back to
direct solving, giving mixed groups), and the curriculum (d4→d6) lets the strategy emerge
where direct solving starts failing.

**Step 3 status: env validated; full run pending billing.** The multi-round memory GRPO env
smoke-tested end-to-end (episodes run 2.6 rounds, accumulate 11.6 notes, reward from
correctness). The full run (`train_tier4_mem.jsonl` d4:150/d5:300/d6:350; group 8, gpb 24,
max_tokens 1600/round, rounds 6, kl 0.05) was launched after a successful auth probe (new key
`tml-eZK22U…` valid) but **died before its first checkpoint when the account re-entered a
billing block (402)** — same balance-exhaustion failure as the Tier-2 episode. Env, launcher,
and curriculum are built/validated/committed; the run re-fires once billing is restored.

## Reproduce

```bash
# steps 1+2 (OpenRouter, temp 0)
python3 scripts/harness_or.py --mode plain  --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_plain_d6.json
python3 scripts/harness_or.py --mode memory --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --out data/runs/gt_rlvr/t4or_harness_d6.json
python3 scripts/harness_or.py --mode decomp --corpus data/runs/gt_rlvr/eval_t3_d6.jsonl --limit 80 --out data/runs/gt_rlvr/t4or_decomp_d6.json
# (same for eval_t3_d7.jsonl)  # step 3: see tinker_grpo_mem.py
```

Artifacts: `data/runs/gt_rlvr/t4or_{plain,harness}_{d6,d7}.json`, `t4or_decomp_d6.json`.
