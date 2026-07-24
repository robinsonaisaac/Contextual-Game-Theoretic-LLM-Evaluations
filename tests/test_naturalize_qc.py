import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from naturalize_traces import qc_ok, _naturalize_one, STYLE_SEEDS
from unittest.mock import Mock


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


def test_naturalize_one_fail_path_includes_style_key():
    """FAIL records must include 'style' key for output contract consistency."""
    # Mock client that always raises (simulating max retries exhausted)
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    item = {
        "prompt": "test prompt",
        "canonical": "test canonical",
        "gold_facts": {"a": "1"},
        "gold_answer": "answer",
        "family": "test_family",
        "style_idx": 2
    }

    result = _naturalize_one(mock_client, item)

    # Verify FAIL record structure
    assert result["ok"] is False
    assert result["completion"] is None
    assert result["prompt"] == "test prompt"
    assert result["family"] == "test_family"
    # KEY: FAIL records must include "style" key
    assert "style" in result
    assert result["style"] == STYLE_SEEDS[2 % len(STYLE_SEEDS)]
