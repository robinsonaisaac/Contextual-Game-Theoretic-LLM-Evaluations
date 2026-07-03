import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tinker_eval import score


def _row(ans):
    return {"answer": ans}


def test_ledger_answer_tail_correct():
    text = "LEDGER\nNOTE: a = 1\nANSWER: 42"
    assert score("ledger", text, _row("42")) == (True, True)


def test_ledger_accepts_answer_tag():
    assert score("ledger", "reasoning...\n<answer>7</answer>", _row("7")) == (True, True)


def test_ledger_normalizes_parens_and_case():
    assert score("ledger", "ANSWER: (A)", _row("a"))[0] is True


def test_ledger_wrong_and_unparsed():
    assert score("ledger", "ANSWER: 5", _row("42")) == (False, True)
    assert score("ledger", "no final answer here", _row("42")) == (False, False)
