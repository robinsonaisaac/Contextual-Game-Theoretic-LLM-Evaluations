# tests/test_invariant_judge_parse.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tier5_attribution import _parse_judge


def test_parse_valid_json():
    raw = ('{"externalizes": true, "grounds_steps": true, "small_steps": false, '
           '"restates": false, "reads_off_answer": true, "uses_ledger": true}')
    d = _parse_judge(raw)
    assert d["uses_ledger"] is True and d["small_steps"] is False


def test_parse_garbage_defaults_false():
    d = _parse_judge("not json")
    assert all(v is False for v in d.values())
    assert set(d) == {"externalizes", "grounds_steps", "small_steps",
                      "restates", "reads_off_answer", "uses_ledger"}


# --- suppression verdict taxonomy (review finding #6) ---
from tier5_attribution import _suppression_verdict


def test_verdict_load_bearing_infamily_collapse():
    # in-family graph: base 0.283, sft 0.900, suppressed 0.050 — the gain collapses
    v = _suppression_verdict(base_acc=0.283, sft_acc=0.900, sft_suppressed=0.050)
    assert v.startswith("load_bearing")


def test_verdict_format_imposition_recovers_to_base():
    # off-family arith: sft harmed, suppression recovers to ~base — knowledge intact
    v = _suppression_verdict(base_acc=0.960, sft_acc=0.304, sft_suppressed=0.948)
    assert v.startswith("format_imposition")


def test_verdict_catastrophic_forgetting_no_recovery():
    # off-family tracking: sft harmed, suppression does NOT recover
    v = _suppression_verdict(base_acc=0.870, sft_acc=0.170, sft_suppressed=0.180)
    assert v.startswith("catastrophic_forgetting")


def test_verdict_partial_mixed():
    v = _suppression_verdict(base_acc=0.900, sft_acc=0.300, sft_suppressed=0.600)
    assert v.startswith("partial")
