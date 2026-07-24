"""New reasoning-operation generators that the existing (deduction-only) suite lacks.

Each is structure-first with an EXACT verifier, knowledge-free, depth-parameterized, and
NON-saturating (difficulty rises monotonically with depth). They add genuinely distinct
cognitive operations so a *breadth* curriculum can be assembled:

  * combinatorial_game   nim_grundy    — multi-heap Nim / Sprague-Grundy (XOR reasoning)
  * inductive_rule       opponent_id   — infer a finite-memory opponent rule from a transcript,
                                          then predict future play (induction from examples)
  * abductive_inference  signal_abduce — observe an action, infer the unique type that
                                          rationalizes it (best-explanation / inverse reasoning)

Pure stdlib (must import under Python 3.9). Answer is always a single integer in <answer>N</answer>.
"""
from __future__ import annotations

import random
import re

_ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def _rng(*key) -> random.Random:
    return random.Random(hash(key) & 0x7FFFFFFF)


# --------------------------------------------------------------------------- #
# Combinatorial game theory — multi-heap Nim, Sprague-Grundy (XOR)
#   depth = number of heaps. The winning move from a *named* heap requires computing the
#   XOR of the OTHER heaps and comparing — genuine Grundy reasoning, non-saturating.
# --------------------------------------------------------------------------- #
def nim_grundy(seed: int, depth: int) -> dict:
    rng = _rng("nim", seed, depth)
    n = depth + 1                                   # depth -> number of heaps (>=2)
    heaps = [rng.randint(1, 15) for _ in range(n)]
    target = 0                                       # XOR of all heaps except heap 1
    for h in heaps[1:]:
        target ^= h
    # To win, reduce heap 1 to `target` (iff target < heap1); stones removed = heap1 - target.
    answer = heaps[0] - target if target < heaps[0] else 0
    listing = ", ".join(f"heap {i+1} has {h} stones" for i, h in enumerate(heaps))
    prompt = (
        f"A game of Nim with {n} heaps: {listing}. Two players alternate; on a turn a player "
        f"removes any positive number of stones from a single heap. The player who removes the "
        f"last stone WINS (normal play). You move first and both play optimally. Considering ONLY "
        f"a first move that takes stones from heap 1, how many stones must you remove from heap 1 "
        f"to leave a winning position? Answer 0 if no winning move is possible from heap 1. "
        f"Reason step by step (use the binary XOR / Sprague-Grundy argument), then answer "
        f"<answer>NUMBER</answer>."
    )
    return {"story_id": f"nim_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "nim_grundy", "op_tags": ["combinatorial_game"]}


# --------------------------------------------------------------------------- #
# Inductive reasoning — identify a finite-memory opponent rule from examples
#   The opponent plays C/D as a deterministic function of the player's last k moves
#   (a 2^k lookup table). A training transcript that covers every k-window reveals the table;
#   the solver must INDUCE it and apply it to a fresh continuation. depth = k.
# --------------------------------------------------------------------------- #
def _de_bruijn(k: int) -> list[int]:
    """Binary de Bruijn sequence B(2,k) as a list of bits (cyclically covers all k-windows)."""
    a = [0] * (k * 2)
    seq: list[int] = []

    def db(t, p):
        if t > k:
            if k % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, 2):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


def opponent_id(seed: int, depth: int) -> dict:
    rng = _rng("opp", seed, max(1, depth))
    k = max(1, depth)                                # memory length; table has 2^k entries
    table = {}
    for bits in range(2 ** k):
        window = tuple((bits >> j) & 1 for j in range(k - 1, -1, -1))
        table[window] = rng.randint(0, 1)
    # my training moves: de Bruijn cycle (covers every k-window) + a k-bit warm-up prefix
    cyc = _de_bruijn(k)
    my_train = cyc + cyc[:k]                          # unrolled so all windows appear as slices
    # ensure variety: opponent responses across the transcript
    opp_train = []
    for t in range(k, len(my_train)):
        opp_train.append(table[tuple(my_train[t - k:t])])
    # the visible transcript starts at the first fully-determined round (t=k)
    vis_my = my_train[k:]
    sym = {0: "C", 1: "D"}
    transcript = ", ".join(f"(you {sym[m]}, them {sym[o]})"
                           for m, o in zip(vis_my, opp_train))
    # test continuation: a fresh sequence of my moves; count opponent cooperations
    M = 6
    # seed the continuation with the last k of the training so the window is defined
    cont_seed = my_train[-k:]
    my_future = [rng.randint(0, 1) for _ in range(M)]
    hist = list(cont_seed) + my_future
    opp_future = [table[tuple(hist[i:i + k])] for i in range(M)]
    answer = sum(1 for o in opp_future if o == 0)    # count of opponent COOPERATIONS (C==0)
    future_str = ", ".join(sym[m] for m in my_future)
    prompt = (
        f"You repeatedly play a 2-action game (each round you and your opponent independently "
        f"play C or D). Your opponent follows a FIXED deterministic rule that depends only on "
        f"your most recent {k} move(s). Here is the full history so far, round by round "
        f"(your move, their move): {transcript}. From these examples, infer the opponent's rule. "
        f"Then suppose that, continuing from the end of that history, you play these next {M} "
        f"moves in order: {future_str}. In how many of those {M} rounds does the opponent play C? "
        f"Reason step by step, then answer <answer>NUMBER</answer>."
    )
    return {"story_id": f"opp_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "opponent_id", "op_tags": ["inductive_rule"]}


# --------------------------------------------------------------------------- #
# Abductive reasoning — infer the hidden type that best explains an observed action
#   N types each pick their payoff-maximizing signal from a known type x signal matrix.
#   Payoffs are regenerated until the type->signal map is a BIJECTION, so the observed
#   signal has a UNIQUE rationalizing type. depth = N. The solver works backward from the
#   effect (observed signal) to the unique cause (type) — best-explanation reasoning.
# --------------------------------------------------------------------------- #
def signal_abduce(seed: int, depth: int) -> dict:
    rng = _rng("abd", seed, depth)
    N = depth + 1                                    # number of types == number of signals (>=2)
    for _ in range(500):
        # payoff[type][signal]
        P = [[rng.randint(0, 20) for _ in range(N)] for _ in range(N)]
        best = []
        ok = True
        for t in range(N):
            row = P[t]
            mx = max(row)
            if row.count(mx) != 1:                   # need a strict argmax (unique optimal signal)
                ok = False
                break
            best.append(row.index(mx))
        if ok and len(set(best)) == N:               # bijection: each signal sent by exactly one type
            break
    else:
        return signal_abduce(seed + 100000, depth)
    observed = rng.randrange(N)                       # the signal we observe
    answer = best.index(observed) + 1                 # the unique type that rationally sends it (1-indexed)
    sig_names = [f"S{j+1}" for j in range(N)]
    rows = "; ".join(
        f"a Type-{t+1} sender earns " + ", ".join(f"{P[t][j]} from {sig_names[j]}" for j in range(N))
        for t in range(N))
    prompt = (
        f"A sender has one of {N} hidden types (Type-1..Type-{N}). Each type chooses exactly one "
        f"signal from {{{', '.join(sig_names)}}} to maximize its own payoff. The payoffs are: "
        f"{rows}. Each type picks the single signal that maximizes its payoff. You OBSERVE the "
        f"sender send signal {sig_names[observed]}. Which type is the sender? Reason step by step "
        f"(work out each type's best signal, then identify which type would send the observed one), "
        f"then answer with the type number as <answer>NUMBER</answer>."
    )
    return {"story_id": f"abd_d{depth}_{seed}", "prompt": prompt, "answer": answer,
            "depth": depth, "family": "signal_abduce", "op_tags": ["abductive_inference"]}


FAMILIES = {"nim_grundy": nim_grundy, "opponent_id": opponent_id, "signal_abduce": signal_abduce}


def extract_answer(text: str):
    m = _ANS.search(text or "")
    return int(m.group(1)) if m else None


def verify(response: str, problem: dict) -> bool:
    pred = extract_answer(response)
    return pred is not None and pred == problem["answer"]
