import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from naturalize_traces import qc_ok


def test_qc_accepts_exact_set_and_answer_any_order():
    gold_facts = {"a": "1", "b": "2"}
    assert qc_ok({("b", "2"), ("a", "1")}, "3", gold_facts, "3") is True


def test_qc_rejects_altered_value():
    assert qc_ok({("a", "1"), ("b", "9")}, "3", {"a": "1", "b": "2"}, "3") is False


def test_qc_rejects_wrong_answer():
    assert qc_ok({("a", "1")}, "4", {"a": "1"}, "3") is False


def test_qc_rejects_missing_or_extra_note():
    assert qc_ok({("a", "1")}, "3", {"a": "1", "b": "2"}, "3") is False
    assert qc_ok({("a", "1"), ("b", "2"), ("c", "9")}, "3", {"a": "1", "b": "2"}, "3") is False
