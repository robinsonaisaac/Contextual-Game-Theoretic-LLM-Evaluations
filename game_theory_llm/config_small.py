# game_theory_llm/config.py
"""Configuration settings and constants."""

# Available Topics
TOPICS = [
    "global politics in the 21st century",
    "social or casual events"
]

# World Building Guidelines
WORLD_DICT = {
    "real world": """REAL WORLD SCENARIO REQUIREMENTS:

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

    The scenario must be grounded in verifiable real-world facts and relationships."""
}

# Actor Types
ACTOR_TYPES = {
    "allies": {
        "description": "Characters with positive relationships",
        "types": ["close_allies", "strategic_partners", "friendly_associates"]
    }
}