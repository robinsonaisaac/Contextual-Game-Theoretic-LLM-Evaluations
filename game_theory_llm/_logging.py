# game_theory_llm/_logging.py
"""Logger factory for the game_theory_llm package.

Library code must NOT call logging.basicConfig() -- that is the caller's
responsibility.  Every module should use:

    from ._logging import get_logger
    logger = get_logger(__name__)
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. No handlers are attached."""
    return logging.getLogger(name)
