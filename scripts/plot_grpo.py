"""Plot the GRPO training curve over time from a run's metrics.jsonl.

GRPO has no scalar supervised loss; the optimized objective is REWARD, so the
training curve is reward-vs-batch (plus KL-to-reference and policy entropy as
health metrics). Re-runnable while training is live.

Usage: python3 scripts/plot_grpo.py [--run data/runs/gt_rlvr/tier1_30b]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="data/runs/gt_rlvr/tier1_30b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    recs = [json.loads(l) for l in (Path(args.run) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    if not recs:
        print("no metrics yet"); return
    x = [r.get("progress/batch", i) for i, r in enumerate(recs)]
    rew = [r.get("env/all/reward/total") for r in recs]
    kl = [r.get("kl_policy_base") for r in recs]
    ent = [r.get("optim/entropy") for r in recs]

    def smooth(y, k=3):
        out = []
        for i in range(len(y)):
            w = [v for v in y[max(0, i - k + 1):i + 1] if v is not None]
            out.append(sum(w) / len(w) if w else None)
        return out

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(x, rew, ".", alpha=0.4, color="#2166ac")
    ax[0].plot(x, smooth(rew), color="#2166ac", lw=2)
    ax[0].set_title("reward (GRPO objective)"); ax[0].set_xlabel("batch"); ax[0].grid(alpha=0.3)
    ax[1].plot(x, kl, color="#b2182b", lw=2)
    ax[1].set_title("KL to reference (anchor)"); ax[1].set_xlabel("batch"); ax[1].grid(alpha=0.3)
    ax[2].plot(x, ent, color="#1a9850", lw=2)
    ax[2].set_title("policy entropy"); ax[2].set_xlabel("batch"); ax[2].grid(alpha=0.3)
    done = recs[-1].get("progress/done_frac", 0)
    fig.suptitle(f"GRPO @ {Path(args.run).name} — {len(recs)} iters, {done*100:.0f}% "
                 f"(reward {rew[0]:.3f} -> {rew[-1]:.3f})")
    fig.tight_layout()
    out = args.out or f"{args.run}/grpo_curve.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"reward {rew[0]:.3f} -> {rew[-1]:.3f} | KL {kl[-1]:.4f} | ent {ent[-1]:.3f} | "
          f"{len(recs)} iters ({done*100:.0f}%) -> {out}")


if __name__ == "__main__":
    main()
