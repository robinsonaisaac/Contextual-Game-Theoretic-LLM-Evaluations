# game_theory_llm/config.py
"""Unified configuration with presets.

Replaces config.py, config_small.py, and config_holdout.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical constants (defined ONCE)
# ---------------------------------------------------------------------------

WORLD_DESCRIPTIONS: Dict[str, str] = {
    "real_world": """REAL WORLD SCENARIO REQUIREMENTS:

    MANDATORY ELEMENTS:
    1. Use only verifiable historical or current real-world entities:
      - Actual companies (e.g., Apple, Microsoft, Tesla)
      - Real countries and their genuine diplomatic relationships
      - Existing organizations (UN, NATO, WHO, etc.)
      - Historical or current political figures
      - Documented business partnerships or rivalries

    RELATIONSHIP GUIDELINES:
    - Only reference documented relationships (allies, competitors, partners)
    - Include specific time periods for historical relationships
    - Consider current geopolitical alliances and tensions
    - Reference actual trade agreements or conflicts
    - Use documented business partnerships or market competitions

    AUTHENTICITY RULES:
    - All events must be based on real historical or current situations
    - Economic relationships must reflect actual market dynamics
    - Military relationships must align with known alliances/conflicts
    - Cultural connections must be based on documented interactions
    - Technological relationships must reflect actual partnerships/competition

    FORBIDDEN ELEMENTS:
    - No fictional organizations or entities
    - No speculative relationships
    - No alternative history scenarios
    - No hypothetical future developments
    - No composite or merged real entities

    The scenario must be grounded in verifiable real-world facts and relationships.""",

    "imaginary_world": """FICTIONAL WORLD-BUILDING REQUIREMENTS:

    WORLD FOUNDATION:
    1. Create a unique world with:
      - Distinctive name and history
      - Original geography and climate systems
      - Unique political structures
      - Original cultural frameworks

    ENTITY CREATION GUIDELINES:
    1. Organizations/Nations:
      - Invent distinctive names
      - Create unique organizational structures
      - Develop original political systems
      - Design innovative power hierarchies
      - Establish clear inter-entity relationships

    2. Cultural Elements:
      - Design unique customs and traditions
      - Create original languages or dialects
      - Develop distinct social hierarchies
      - Invent unique technologies or magic systems
      - Establish original religious or philosophical systems

    RELATIONSHIP DYNAMICS:
    1. Power Structures:
      - Create complex alliance systems
      - Design unique conflict resolution methods
      - Develop original resource distribution systems
      - Establish innovative diplomatic protocols

    2. World Mechanics:
      - Design original natural laws or magic systems
      - Create unique resource dynamics
      - Develop innovative communication methods
      - Establish original transportation systems
      - Invent unique economic exchanges

    AUTHENTICITY RULES:
    - Maintain internal consistency
    - Develop believable motivations for all entities
    - Ensure technological/magical systems follow established rules
    - Build coherent historical backgrounds

    FORBIDDEN ELEMENTS:
    - No direct copying of real-world entities
    - No Earth-specific references
    - No existing historical parallels
    - No current political systems
    - No real-world organizations""",
}

ACTOR_TYPES: Dict[str, dict] = {
    "allies": {
        "description": "Characters with positive relationships",
        "types": ["close_allies", "strategic_partners", "friendly_associates"],
    },
    "enemies": {
        "description": "Characters with negative relationships",
        "types": ["sworn_enemies", "competitors", "adversaries"],
    },
    "neutral": {
        "description": "Characters with neutral relationships",
        "types": ["neutral_strangers", "distant_acquaintances"],
    },
}

ALL_TOPICS: List[str] = [
    "global politics in the 5th century",
    "global politics in the 20th century",
    "global politics in the 21st century",
    "United States politics in 2020",
    "international business",
    "business within the United States",
    "politics",
    "business",
    "social or casual events",
    "sporting events",
    # Holdout topics
    "Environmental policy negotiations",
    "Online marketplaces",
    "Cybersecurity",
]


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Describes a particular experiment slice."""
    topics: List[str]
    world_types: List[str]
    actor_types: List[str]

    def validate(self) -> None:
        """Raise ``ValueError`` for unknown topics / world types / actor types."""
        for t in self.topics:
            if t not in ALL_TOPICS:
                raise ValueError(f"Unknown topic: {t!r}")
        for w in self.world_types:
            if w not in WORLD_DESCRIPTIONS:
                raise ValueError(f"Unknown world type: {w!r}")
        for a in self.actor_types:
            if a not in ACTOR_TYPES:
                raise ValueError(f"Unknown actor type: {a!r}")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, ExperimentConfig] = {
    "full": ExperimentConfig(
        topics=[
            "global politics in the 5th century",
            "global politics in the 20th century",
            "global politics in the 21st century",
            "United States politics in 2020",
            "international business",
            "business within the United States",
            "politics",
            "business",
            "social or casual events",
            "sporting events",
        ],
        world_types=["real_world", "imaginary_world"],
        actor_types=["allies", "enemies", "neutral"],
    ),
    "small": ExperimentConfig(
        topics=[
            "global politics in the 21st century",
            "social or casual events",
        ],
        world_types=["real_world"],
        actor_types=["allies"],
    ),
    "holdout": ExperimentConfig(
        topics=[
            "Environmental policy negotiations",
            "Online marketplaces",
            "Cybersecurity",
        ],
        world_types=["real_world", "imaginary_world"],
        actor_types=["allies", "enemies", "neutral"],
    ),
}


def get_config(preset: str = "full") -> ExperimentConfig:
    """Return a validated ``ExperimentConfig`` for the given preset name.

    Raises ``ValueError`` if *preset* is not recognised.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(PRESETS)}")
    cfg = PRESETS[preset]
    cfg.validate()
    return cfg
