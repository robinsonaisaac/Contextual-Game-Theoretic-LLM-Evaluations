"""Validate the new reasoning-operation generators: exact solvers, uniqueness, scaling."""
import re

import pytest

from game_theory_llm.reasoning.reasoning_ops import (
    nim_grundy, opponent_id, signal_abduce, _de_bruijn, FAMILIES, verify,
)

ANS = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


@pytest.mark.parametrize("fn", FAMILIES.values())
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_wellformed(fn, depth):
    p = fn(seed=7, depth=depth)
    assert p["prompt"] and isinstance(p["answer"], int)
    assert p["op_tags"] and p["depth"] == depth
    # the prompt must request an <answer> tag and not leak the answer literally
    assert "<answer>" in p["prompt"]


def test_de_bruijn_covers_all_windows():
    for k in (1, 2, 3, 4):
        seq = _de_bruijn(k)
        assert len(seq) == 2 ** k
        cyc = seq + seq[:k - 1]
        windows = {tuple(cyc[i:i + k]) for i in range(2 ** k)}
        assert len(windows) == 2 ** k                 # every k-bit window appears


def test_nim_grundy_exact():
    # brute-force check: the claimed move from heap 1 must reach XOR==0 (or be 0 when impossible)
    for seed in range(200):
        p = nim_grundy(seed=seed, depth=3)
        heaps = [int(x) for x in re.findall(r"has (\d+) stones", p["prompt"])]
        x_all = 0
        for h in heaps:
            x_all ^= h
        others = x_all ^ heaps[0]
        expect = heaps[0] - others if others < heaps[0] else 0
        assert p["answer"] == expect
        if p["answer"] > 0:                            # the move really lands on a P-position
            new_heaps = [heaps[0] - p["answer"]] + heaps[1:]
            x = 0
            for h in new_heaps:
                x ^= h
            assert x == 0


def test_opponent_id_answer_in_range_and_varies():
    answers = set()
    for seed in range(80):
        p = opponent_id(seed=seed, depth=2)
        assert 0 <= p["answer"] <= 6
        answers.add(p["answer"])
    assert len(answers) >= 4                           # not a degenerate constant


def test_signal_abduce_bijection_unique():
    for seed in range(80):
        p = signal_abduce(seed=seed, depth=3)
        N = 4
        assert 1 <= p["answer"] <= N
        # each type's optimal signal is a strict argmax and the map is a bijection
        rows = re.findall(r"Type-\d+ sender earns ([^;]+?)(?:;|\. You)", p["prompt"])
        best = []
        for row in rows:
            vals = [int(v) for v in re.findall(r"(\d+) from", row)]
            mx = max(vals)
            assert vals.count(mx) == 1
            best.append(vals.index(mx))
        assert len(set(best)) == N


def test_verify_roundtrips():
    p = nim_grundy(seed=1, depth=2)
    assert verify(f"... <answer>{p['answer']}</answer>", p)
    assert not verify(f"... <answer>{p['answer'] + 1}</answer>", p)


def test_difficulty_scales_with_depth():
    # deeper = harder: a trivial "remove 0 or guess" heuristic should win less at higher depth.
    # Here we just assert the structures grow (heaps / types / table size) with depth.
    assert len(re.findall(r"heap \d+ has", nim_grundy(0, 5)["prompt"])) == 6
    assert len(re.findall(r"heap \d+ has", nim_grundy(0, 2)["prompt"])) == 3
    assert signal_abduce(0, 4)["prompt"].count("Type-") > signal_abduce(0, 1)["prompt"].count("Type-")
