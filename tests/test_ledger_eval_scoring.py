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


def test_dyck_exact_match_no_bracket_stripping():
    # distinct bracket answers must NOT compare equal (finding #5: strip(".()") collapsed
    # ")" vs "()" and "())" vs ")))" to the same normalized "")
    assert score("dyck", "reasoning...\nANSWER: )", _row(")")) == (True, True)
    assert score("dyck", "reasoning...\nANSWER: ()", _row("()")) == (True, True)
    assert score("dyck", "reasoning...\nANSWER: )", _row("()"))[0] is False
    assert score("dyck", "reasoning...\nANSWER: ())", _row(")))"))[0] is False


def test_dyck_accepts_answer_tag_too():
    assert score("dyck", "work\n<answer>]]></answer>", _row("]]>")) == (True, True)


def test_dyck_unparsed_when_no_tail():
    assert score("dyck", "no final answer here", _row(")"))  == (False, False)
