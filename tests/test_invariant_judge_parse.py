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
