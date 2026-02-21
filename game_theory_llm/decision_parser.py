# game_theory_llm/decision_parser.py
"""Decision extraction from LLM responses.

Fixes bugs #2, #3, #4 from the original code:
  - Pattern 3 regex now matches * and ** (was matching & due to typo)
  - All patterns normalize to A/B
  - Pattern 5 (#/&) restored (was commented out)
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ._logging import get_logger

logger = get_logger(__name__)


@dataclass
class DecisionPattern:
    """A regex + mapping pair for extracting decisions."""
    regex: str
    mapping: Dict[str, str]


DEFAULT_PATTERNS: List[DecisionPattern] = [
    # Pattern 1: A/B
    DecisionPattern(
        regex=r'<decision>\s*([AB])\s*</decision>',
        mapping={'A': 'A', 'B': 'B'},
    ),
    # Pattern 2: 1/2
    DecisionPattern(
        regex=r'<decision>\s*([12])\s*</decision>',
        mapping={'1': 'A', '2': 'B'},
    ),
    # Pattern 3: * / ** (bug #2 & #3 fixed)
    DecisionPattern(
        regex=r'<decision>\s*(\*{1,2})\s*</decision>',
        mapping={'*': 'A', '**': 'B'},
    ),
    # Pattern 4: yellow/green
    DecisionPattern(
        regex=r'<decision>\s*(yellow|green)\s*</decision>',
        mapping={'yellow': 'A', 'green': 'B'},
    ),
    # Pattern 5: #/& (restored — was commented out)
    DecisionPattern(
        regex=r'<decision>\s*([#&])\s*</decision>',
        mapping={'#': 'A', '&': 'B'},
    ),
]


def extract_decision(
    text: str,
    patterns: Optional[List[DecisionPattern]] = None,
) -> Optional[str]:
    """Extract a decision (A or B) from *text* using the given patterns.

    Parameters
    ----------
    text : str
        The LLM response text to parse.
    patterns : list[DecisionPattern] | None
        Patterns to try, in order.  Defaults to ``DEFAULT_PATTERNS``.

    Returns
    -------
    str | None
        ``'A'`` or ``'B'`` if a valid decision is found, otherwise ``None``.
    """
    if patterns is None:
        patterns = DEFAULT_PATTERNS

    for pat in patterns:
        match = re.search(pat.regex, text)
        if match:
            raw = match.group(1)
            mapped = pat.mapping.get(raw)
            if mapped:
                logger.debug("Extracted decision: %s -> mapped to: %s", raw, mapped)
                return mapped

    logger.warning("Could not parse <decision> from response")
    return None
