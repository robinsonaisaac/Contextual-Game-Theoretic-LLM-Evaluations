# tests/test_decision_parser.py
"""Tests for game_theory_llm.decision_parser."""

import pytest

from game_theory_llm.decision_parser import (
    DEFAULT_PATTERNS,
    DecisionPattern,
    extract_decision,
)


class TestPattern1_AB:
    def test_decision_a(self):
        assert extract_decision("<decision>A</decision>") == "A"

    def test_decision_b(self):
        assert extract_decision("<decision>B</decision>") == "B"

    def test_with_whitespace(self):
        assert extract_decision("<decision>  A  </decision>") == "A"

    def test_case_sensitive(self):
        # lowercase 'a' should NOT match pattern 1
        assert extract_decision("<decision>a</decision>") is None


class TestPattern2_Numeric:
    def test_decision_1_maps_to_a(self):
        assert extract_decision("<decision>1</decision>") == "A"

    def test_decision_2_maps_to_b(self):
        assert extract_decision("<decision>2</decision>") == "B"


class TestPattern3_Stars:
    def test_single_star_maps_to_a(self):
        assert extract_decision("<decision>*</decision>") == "A"

    def test_double_star_maps_to_b(self):
        assert extract_decision("<decision>**</decision>") == "B"


class TestPattern4_Colors:
    def test_yellow_maps_to_a(self):
        assert extract_decision("<decision>yellow</decision>") == "A"

    def test_green_maps_to_b(self):
        assert extract_decision("<decision>green</decision>") == "B"


class TestPattern5_Symbols:
    def test_hash_maps_to_a(self):
        assert extract_decision("<decision>#</decision>") == "A"

    def test_ampersand_maps_to_b(self):
        assert extract_decision("<decision>&</decision>") == "B"


class TestEdgeCases:
    def test_no_decision_tag(self):
        assert extract_decision("I choose option A") is None

    def test_empty_decision_tag(self):
        assert extract_decision("<decision></decision>") is None

    def test_invalid_value(self):
        assert extract_decision("<decision>X</decision>") is None

    def test_nested_tags(self):
        text = "<analysis>blah <decision>A</decision> blah</analysis>"
        assert extract_decision(text) == "A"

    def test_surrounded_by_other_text(self):
        text = "Here is my analysis. I think <decision>B</decision> is the best choice."
        assert extract_decision(text) == "B"

    def test_first_pattern_wins(self):
        text = "<decision>A</decision> ... <decision>B</decision>"
        assert extract_decision(text) == "A"

    def test_custom_patterns(self):
        custom = [
            DecisionPattern(
                regex=r"<decision>\s*(yes|no)\s*</decision>",
                mapping={"yes": "A", "no": "B"},
            )
        ]
        assert extract_decision("<decision>yes</decision>", patterns=custom) == "A"
        assert extract_decision("<decision>no</decision>", patterns=custom) == "B"
        # default pattern should not work with custom
        assert extract_decision("<decision>A</decision>", patterns=custom) is None
