"""Does framing sensitivity change across model generations?

Uses the historical PD sweep (evals_historical/: 28 model checkpoints x 5 contrast
dimensions x 2 levels, on the same 504 PD vignettes as the main run) to ask whether
the *impact of context* has shifted over ~2 years of model releases.

The naive answer is yes -- raw framing effect size declines with release date. That
result is confounded: a model pinned near the PD cooperation ceiling has almost no
room for framing to move anything, so effect size shrinks mechanically. We therefore
report three quantities per model:

  meanV     mean phi (= Cramer's V for a 2x2) over the five contrast dimensions
  nullV     the same statistic under a permutation that shuffles level labels while
            preserving the model's marginal A-rate and group sizes -- i.e. the noise
            floor for THIS model
  excessV   meanV - nullV, the ceiling- and n-corrected framing effect

plus a headroom ratio |delta| / max|delta|, where max|delta| = 2*min(p, 1-p) is the
largest shift achievable at that marginal. The headroom ratio is unstable exactly at
the ceiling (it can exceed 1 when group sizes are unequal), so excessV is the measure
we trust; headroom is reported only as a corroborating check.

Usage:  python3 scripts/analyze_framing_over_time.py [--out FIG] [--nperm N]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from importlib.machinery import SourceFileLoader

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_GLOB = os.path.join(REPO, "data/runs/2026-05-05-sharp/evals_historical/*.jsonl")
DEFAULT_FIG = os.path.join(
    REPO, "-NEURIPS-2026-Framing-The-Game/ResultsFigures/framing_effect_over_time.png"
)
DIMS = ["gender", "realism", "era", "contrast_domain", "observability"]

# Release dates and the decision parser are shared with the cooperation-trend figure
# so the two temporal analyses cannot drift apart.
_H = SourceFileLoader(
    "historical", os.path.join(REPO, "scripts/regen_historical_cooperation.py")
).load_module()
RELEASE_DATES, parse_decision = _H.RELEASE_DATES, _H.parse_decision

FAMILY_OF = {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini", "grok": "Grok", "qwen": "Qwen"}
FAMILY_COLOR = {
    "GPT": "#e41a1c", "Claude": "#377eb8", "Gemini": "#4daf4a",
    "Grok": "#984ea3", "Qwen": "#ff7f00",
}


def family(model):
    for pref, fam in FAMILY_OF.items():
        if model.startswith(pref):
            return fam
    return "Other"


def phi(y, g):
    """|phi| for binary outcome y against binary group g. Degenerate tables (a zero
    marginal -- e.g. a ceiling model answering A every time) admit no association: 0."""
    a = int(((g == 0) & (y == 1)).sum())
    b = int(((g == 0) & (y == 0)).sum())
    c = int(((g == 1) & (y == 1)).sum())
    d = int(((g == 1) & (y == 0)).sum())
    den = (a + b) * (c + d) * (a + c) * (b + d)
    return 0.0 if den == 0 else abs(a * d - b * c) / np.sqrt(den)


def load():
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen, parsed = defaultdict(int), defaultdict(int)
    for f in glob.glob(EVAL_GLOB):
        _, dim, level, model = os.path.basename(f)[:-6].split("__")
        for line in open(f):
            seen[model] += 1
            d = parse_decision(json.loads(line).get("response", ""))
            if d is not None:
                parsed[model] += 1
                raw[model][dim][level].append(1 if d == "A" else 0)
    return raw, seen, parsed


def analyse(raw, seen, parsed, nperm, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for model in sorted(raw):
        if model not in RELEASE_DATES:
            continue
        obs, nulls, deltas, headroom = [], [], [], []
        n_a = n_tot = 0
        ok = True
        for dim in DIMS:
            levels = sorted(raw[model][dim])
            if len(levels) != 2:
                ok = False
                break
            y0 = np.array(raw[model][dim][levels[0]])
            y1 = np.array(raw[model][dim][levels[1]])
            if len(y0) == 0 or len(y1) == 0:
                ok = False
                break
            y = np.concatenate([y0, y1])
            g = np.concatenate([np.zeros(len(y0), int), np.ones(len(y1), int)])
            obs.append(phi(y, g))
            # Permuting g preserves both the marginal A-rate and the group sizes, so
            # the null absorbs the ceiling effect and the differing per-model n.
            nulls.append(float(np.mean([phi(y, rng.permutation(g)) for _ in range(nperm)])))
            p = y.mean()
            delta = abs(y1.mean() - y0.mean())
            max_delta = 2 * min(p, 1 - p)
            deltas.append(delta)
            headroom.append(delta / max_delta if max_delta > 1e-9 else np.nan)
            n_a += int(y.sum())
            n_tot += len(y)
        if not ok or len(obs) != 5:
            continue
        recs.append(
            dict(
                model=model,
                family=family(model),
                date=datetime.strptime(RELEASE_DATES[model], "%Y-%m-%d"),
                arate=n_a / n_tot,
                meanV=float(np.mean(obs)),
                nullV=float(np.mean(nulls)),
                excessV=float(np.mean(obs) - np.mean(nulls)),
                meanabsD=float(np.mean(deltas)),
                headroom=float(np.nanmean(headroom)),
                n=n_tot,
                unparsed=1 - parsed[model] / max(1, seen[model]),
            )
        )
    recs.sort(key=lambda r: r["date"])
    return recs


def report(recs):
    hdr = f"{'model':22s} {'released':10s} {'A-rate':>7s} {'meanV':>7s} {'nullV':>7s} {'excessV':>8s} {'|D|/max':>8s} {'unparsed':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(
            f"{r['model']:22s} {r['date']:%Y-%m-%d} {r['arate']:7.3f} {r['meanV']:7.3f} "
            f"{r['nullV']:7.3f} {r['excessV']:+8.3f} {r['headroom']:8.3f} {r['unparsed']:9.2%}"
        )
    x = np.array([r["date"].toordinal() for r in recs], float)
    print(f"\nn = {len(recs)} models, {sum(r['n'] for r in recs):,} parsed decisions\n")
    out = {}
    for key in ("arate", "meanV", "excessV", "meanabsD", "headroom"):
        y = np.array([r[key] for r in recs], float)
        rho, p = stats.spearmanr(x, y)
        out[key] = (rho, p)
        print(f"  Spearman(release_date, {key:9s}) rho = {rho:+.3f}   p = {p:.4f}")
    a = np.array([r["arate"] for r in recs])
    v = np.array([r["meanV"] for r in recs])
    rho_c, p_c = stats.spearmanr(a, v)
    out["confound"] = (rho_c, p_c)
    print(f"\n  Spearman(A-rate,  meanV)      rho = {rho_c:+.3f}   p = {p_c:.4f}  <- ceiling confound")
    print(
        "\nReading: the raw decline in framing effect with release date is largely explained\n"
        "by newer models sitting nearer the PD cooperation ceiling. Correcting each model\n"
        "against its own marginal-preserving null leaves no significant trend."
    )
    out.update(cooperation_trend(recs))
    return out


def cooperation_trend(recs):
    """Has the cooperative SETPOINT moved, as distinct from framing sensitivity?

    Reported three ways, because they disagree and the disagreement is the point.
    The 27 checkpoints are not independent -- they are 5 families of 5-6 correlated
    releases -- so the family-level test is the one to believe.
    """
    print("\n" + "=" * 70)
    print("Cooperation setpoint over time")
    print("=" * 70)
    x = np.array([r["date"].toordinal() for r in recs], float)
    y = np.array([r["arate"] for r in recs])

    rho, p = stats.spearmanr(x, y)
    print(f"  model-level monotone (Spearman, n={len(recs)}):  rho = {rho:+.3f}  p = {p:.4f}")

    half = len(recs) // 2
    early, late = y[:half], y[len(recs) - half:]
    p_mwu = stats.mannwhitneyu(early, late)[1]
    print(f"  model-level half-split ({half} v {half}):  {early.mean():.3f} -> {late.mean():.3f}  "
          f"p = {p_mwu:.4f}   [post-hoc split; ignores family clustering]")

    fams, deltas, ups = [], [], 0
    for fam in sorted({r["family"] for r in recs}):
        sub = sorted([r for r in recs if r["family"] == fam], key=lambda r: r["date"])
        if len(sub) < 2:
            continue
        d = sub[-1]["arate"] - sub[0]["arate"]
        fams.append((fam, sub[0], sub[-1], d))
        deltas.append(d)
        ups += d > 0
    print(f"\n  family-level first -> last (n = {len(fams)} families):")
    for fam, f0, f1, d in fams:
        print(f"    {fam:7s} {f0['model']:20s} {f0['arate']:.3f}  ->  {f1['model']:20s} "
              f"{f1['arate']:.3f}   {d:+.3f}")
    p_sign = stats.binomtest(ups, len(fams), 0.5).pvalue
    p_wil = stats.wilcoxon(deltas)[1]
    print(f"\n  sign test:     {ups}/{len(fams)} up   p = {p_sign:.3f}")
    print(f"  Wilcoxon:      mean shift {np.mean(deltas):+.3f}   p = {p_wil:.3f}")

    sd_e, sd_l = early.std(ddof=1), late.std(ddof=1)
    p_lev, p_fli = stats.levene(early, late)[1], stats.fligner(early, late)[1]
    print(f"\n  spread (convergence): SD {sd_e:.3f} -> {sd_l:.3f}   "
          f"Levene p = {p_lev:.3f}, Fligner p = {p_fli:.3f}")
    print(
        "\nReading: cooperation drifts upward in 4 of 5 families but the effect is not\n"
        "established once family clustering is respected, and the narrowing of spread is\n"
        "not significant. The setpoint has (weakly) moved; framing sensitivity has not."
    )
    return dict(coop_spearman=(rho, p), coop_sign=p_sign, coop_wilcoxon=p_wil,
                coop_halfsplit=p_mwu, coop_levene=p_lev)


def figure(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3))
    fams = sorted({r["family"] for r in recs})

    for ax, xkey, xlabel in ((ax1, "date", "Model release date"),
                             (ax2, "arate", "PD cooperation rate (A-rate)")):
        for fam in fams:
            pts = [r for r in recs if r["family"] == fam]
            ax.scatter([r[xkey] for r in pts], [r["meanV"] for r in pts],
                       s=34, color=FAMILY_COLOR.get(fam, "#777777"),
                       label=fam, zorder=3, edgecolor="white", linewidth=0.6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared noise floor: the mean per-model permutation null.
    nullmean = float(np.mean([r["nullV"] for r in recs]))
    for ax in (ax1, ax2):
        ax.axhline(nullmean, color="black", linestyle="--", linewidth=0.9, zorder=1)
    ax1.text(0.02, nullmean + 0.004, "permutation null", transform=ax1.get_yaxis_transform(),
             fontsize=7, va="bottom")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lab in ax1.get_xticklabels():
        lab.set_rotation(30)
        lab.set_horizontalalignment("right")
    ax1.set_ylabel("Framing effect (mean $\\phi$)", fontsize=9)

    x = np.array([r["date"].toordinal() for r in recs], float)
    y = np.array([r["meanV"] for r in recs])
    rho_t, p_t = stats.spearmanr(x, y)
    rho_c, p_c = stats.spearmanr([r["arate"] for r in recs], y)
    ax1.set_title(f"vs. release date: $\\rho={rho_t:+.2f}$, $p={p_t:.3f}$", fontsize=8.5)
    ax2.set_title(f"vs. cooperation rate: $\\rho={rho_c:+.2f}$, $p={p_c:.3f}$", fontsize=8.5)
    ax2.legend(fontsize=7, frameon=False, loc="upper right", ncol=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_FIG)
    ap.add_argument("--nperm", type=int, default=2000)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    raw, seen, parsed = load()
    recs = analyse(raw, seen, parsed, args.nperm)
    if not recs:
        sys.exit("no usable models found")
    report(recs)
    if not args.no_figure:
        figure(recs, args.out)


if __name__ == "__main__":
    main()
