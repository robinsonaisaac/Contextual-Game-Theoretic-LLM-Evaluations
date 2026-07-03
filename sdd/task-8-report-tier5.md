# Task 8 Report — Tier-5 Gate Eval

**Date:** 2026-07-03  
**Base model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`  
**SFT checkpoint:** `tinker://71664076-e42f-5b01-8ff8-337cec9df9e1:train:0/sampler_weights/tier5_sft`  
**Eval sets:** 12 in-domain sets × 2 models = 24 eval runs; 60 rows each  

---

## Gate Decision

**PROCEED**

Triggered by second PROCEED condition: `mean_trained_sft_acc (0.658) ≥ 0.60 AND mean_extrap_delta (0.071) ≥ 0.05`.

First condition (mean_extrap_delta ≥ 0.15) was NOT met (actual: 0.071).  
SKIP-RL condition (mean_sft ≥ 0.90) was NOT met (actual: 0.465).

---

## Per-Set Results

| Set | Type | base_acc | sft_acc | delta |
|-----|------|----------|---------|-------|
| forward_chain_h60 | trained | 0.200 | **1.000** | +0.800 |
| forward_chain_h90 | trained | 0.033 | **1.000** | +0.967 |
| forward_chain_h130 | EXTRAP | 0.033 | 0.117 | +0.083 |
| graph_search_h60 | trained | 0.317 | **1.000** | +0.683 |
| graph_search_h90 | trained | 0.033 | 0.467 | +0.433 |
| graph_search_h130 | EXTRAP | 0.000 | 0.200 | **+0.200** |
| register_machine_h60 | trained | 0.700 | 0.800 | +0.100 |
| register_machine_h90 | trained | 0.317 | **0.000** | **−0.317** |
| register_machine_h150 | EXTRAP | 0.000 | 0.000 | 0.000 |
| trees_h31 | trained | 0.083 | **0.983** | +0.900 |
| trees_h63 | trained | 0.017 | 0.017 | 0.000 |
| trees_h127 | EXTRAP | 0.000 | 0.000 | 0.000 |

---

## Per-Family Aggregates

| Family | mean_base (trained) | mean_sft (trained) | mean_base (EXTRAP) | mean_sft (EXTRAP) | EXTRAP delta |
|--------|---------------------|--------------------|--------------------|-------------------|--------------|
| forward_chain | 0.117 | 1.000 | 0.033 | 0.117 | +0.083 |
| graph_search | 0.175 | 0.733 | 0.000 | 0.200 | +0.200 |
| register_machine | 0.508 | 0.400 | 0.000 | 0.000 | 0.000 |
| trees | 0.050 | 0.500 | 0.000 | 0.000 | 0.000 |

---

## Extrapolated-Horizon Summary

| EXTRAP set | horizon | base_acc | sft_acc | delta |
|------------|---------|----------|---------|-------|
| forward_chain_h130 | 130 | 0.033 | 0.117 | +0.083 |
| graph_search_h130 | 130 | 0.000 | 0.200 | +0.200 |
| register_machine_h150 | 150 | 0.000 | 0.000 | 0.000 |
| trees_h127 | 127 | 0.000 | 0.000 | 0.000 |
| **mean** | | 0.008 | 0.079 | **+0.071** |

---

## Gate Metrics

| Metric | Value | Threshold | Met? |
|--------|-------|-----------|------|
| mean_sft_acc (all 12 sets) | 0.465 | ≥ 0.90 → SKIP-RL | No |
| mean_extrap_delta (4 EXTRAP sets) | 0.071 | ≥ 0.15 → PROCEED | No |
| mean_trained_sft_acc (8 trained sets) | 0.658 | ≥ 0.60 (AND delta ≥ 0.05) → PROCEED | **Yes** |
| mean_extrap_delta | 0.071 | ≥ 0.05 (AND trained ≥ 0.60) | **Yes** |

**Decision: PROCEED** — second condition met.

---

## Per-Family Anomalies

### register_machine — catastrophic SFT regression at h90
- base h90 = 0.317, sft h90 = **0.000** (parse_rate = 0.000 for SFT)
- base h150 = 0.000, sft h150 = 0.000
- The SFT checkpoint produces no parseable ANSWER: output for register_machine at horizons ≥ 90, despite the base model generating output (parse_rate = 0.417 at h90)
- At h60, base = 0.700 (the highest base accuracy of any set), SFT = 0.800 — ceiling effect

### trees — bimodal SFT performance
- h31 trained: sft = **0.983** (parse_rate = 1.000) — excellent
- h63 trained: sft = 0.017 (parse_rate = 1.000) — fully formatted but entirely wrong
- h127 EXTRAP: sft = 0.000 (parse_rate = 0.000)
- The SFT model outputs ANSWER: format perfectly at h63 but the answers are wrong, suggesting the formatting was learned but the reasoning was not transferred

### forward_chain and graph_search — strong SFT transfer
- Trained horizons (h60, h90): SFT achieves near-perfect accuracy (1.000, 1.000, 1.000, 0.467)
- EXTRAP horizons: partial transfer (0.117 and 0.200 vs base 0.033 and 0.000)
- graph_search_h130 is the best EXTRAP result (delta = +0.200)

---

## Smoke Test (from Task 8 Phase 1)

Corpus: `eval_indomain_forward_chain_h60.jsonl` (10 rows), base model  
Result: accuracy = 0.300, parse_rate = 0.700  
Example extractions:
- `chain_h60_s100000`: `ANSWER: yes` matched gold `yes` → correct=True, parsed=True  
- `chain_h60_s100001`: `ANSWER: no` vs gold `yes` → correct=False, parsed=True  

---

## Commit

SHA `45d4411` — "feat(tier5): ledger eval kind (ANSWER tail) + Phase-1 gate driver"  
Files: `scripts/tinker_eval.py` (ledger eval kind, lazy imports), `scripts/tier5_gate.py`, `tests/test_ledger_eval_scoring.py` (4 tests pass)
