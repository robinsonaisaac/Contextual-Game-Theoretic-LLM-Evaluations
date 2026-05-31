"""Unit tests for GSM8k + HumanEval capability scoring (no GPU/network)."""

from game_theory_llm.capability_scoring import (
    extract_gsm8k_answer, gold_gsm8k_answer, gsm8k_correct,
    extract_code, humaneval_passes,
)


# --- GSM8k numeric extraction ------------------------------------------------

def test_extract_answer_tag():
    assert extract_gsm8k_answer("blah blah <answer>42</answer>") == 42.0

def test_extract_answer_tag_with_commas_and_dollar():
    assert extract_gsm8k_answer("So <answer>$1,234</answer>") == 1234.0

def test_extract_boxed():
    assert extract_gsm8k_answer(r"therefore \boxed{18}") == 18.0

def test_extract_last_number_fallback():
    assert extract_gsm8k_answer("step1 = 5, step2 = 7, total is 12") == 12.0

def test_extract_none_when_no_number():
    assert extract_gsm8k_answer("no digits here") is None

def test_gold_parsing():
    assert gold_gsm8k_answer("Reasoning...\n#### 72") == 72.0
    assert gold_gsm8k_answer("x\n#### 3,500") == 3500.0

def test_gsm8k_correct_true_false():
    assert gsm8k_correct("the answer is <answer>72</answer>", 72) is True
    assert gsm8k_correct("the answer is <answer>71</answer>", 72) is False
    # raw gold strings ("#### 72") are normalized too -> still matches
    assert gsm8k_correct("<answer>72</answer>", "#### 72") is True

def test_gsm8k_correct_with_gold_float():
    assert gsm8k_correct("<answer>72.0</answer>", 72.0) is True


# --- HumanEval code extraction + execution -----------------------------------

def test_extract_code_block():
    trace = "Here:\n```python\ndef f(x):\n    return x+1\n```\ndone"
    assert "def f(x):" in extract_code(trace)
    assert "done" not in extract_code(trace)

def test_extract_code_no_block_returns_all():
    assert extract_code("def g(): return 1").strip() == "def g(): return 1"

def test_humaneval_passes_correct_solution():
    prompt = "def add(a, b):\n    "
    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n"
    trace = "```python\ndef add(a, b):\n    return a + b\n```"
    assert humaneval_passes(trace, prompt, test, "add") is True

def test_humaneval_fails_wrong_solution():
    prompt = "def add(a, b):\n    "
    test = "def check(candidate):\n    assert candidate(2, 3) == 5\n"
    trace = "```python\ndef add(a, b):\n    return a - b\n```"
    assert humaneval_passes(trace, prompt, test, "add") is False

def test_humaneval_continuation_uses_prompt_stub():
    # model returns only the body continuation (no def line)
    prompt = "def square(x):\n"
    test = "def check(candidate):\n    assert candidate(4) == 16\n"
    trace = "```python\n    return x * x\n```"
    assert humaneval_passes(trace, prompt, test, "square") is True

def test_humaneval_timeout_is_failure():
    prompt = "def loop():\n"
    test = "def check(candidate):\n    candidate()\n"
    trace = "```python\ndef loop():\n    while True:\n        pass\n```"
    assert humaneval_passes(trace, prompt, test, "loop", timeout=2.0) is False
