"""Classic 42-territory Hasbro Risk map — pure data table.

This module is a *data-only* deliverable (Unit S). It defines the standard
Risk board so that the Risk game (``play/games/risk_lite.py``, Unit 3) can be
upgraded from its toy 6-territory map to the full board without any of the
game logic living here.

Exposed symbols (frozen interface — see spec §8):

- ``TERRITORIES: list[str]``  — length 42; the list index IS the territory id.
- ``ADJ: dict[int, frozenset[int]]`` — canonical undirected adjacency,
  INCLUDING the standard sea connections (Alaska<->Kamchatka,
  Greenland<->Iceland, Brazil<->North Africa, Western Europe<->North Africa,
  Southern Europe<->Egypt/North Africa, East Africa<->Middle East,
  Siam<->Indonesia, etc.). Every edge appears in both directions.
- ``CONTINENTS: dict[str, tuple[frozenset[int], int]]`` — the 6 continents
  mapped to ``(member territory ids, reinforcement bonus)``. Bonuses:
  North America 5, South America 2, Europe 5, Africa 3, Asia 7, Australia 2.
- ``SET_VALUES: list[int] = [4, 6, 8, 10, 12, 15]`` — escalating Risk-card
  set trade-in values.
- ``SET_INCREMENT: int = 5`` — each set traded after the sixth is worth the
  previous value + 5 (i.e. 20, 25, 30, ...).

The board is the standard Hasbro layout. Adjacency is symmetric and the
whole graph is connected (every territory is reachable from every other).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple


# ---------------------------------------------------------------------------
# Territories.  Grouped by continent for readability; the flat ``TERRITORIES``
# list index is the canonical territory id used everywhere else.
# ---------------------------------------------------------------------------

# North America (ids 0-8)
ALASKA = 0
NORTHWEST_TERRITORY = 1
GREENLAND = 2
ALBERTA = 3
ONTARIO = 4
QUEBEC = 5
WESTERN_UNITED_STATES = 6
EASTERN_UNITED_STATES = 7
CENTRAL_AMERICA = 8

# South America (ids 9-12)
VENEZUELA = 9
PERU = 10
BRAZIL = 11
ARGENTINA = 12

# Europe (ids 13-19)
ICELAND = 13
SCANDINAVIA = 14
GREAT_BRITAIN = 15
NORTHERN_EUROPE = 16
WESTERN_EUROPE = 17
SOUTHERN_EUROPE = 18
UKRAINE = 19

# Africa (ids 20-25)
NORTH_AFRICA = 20
EGYPT = 21
EAST_AFRICA = 22
CONGO = 23
SOUTH_AFRICA = 24
MADAGASCAR = 25

# Asia (ids 26-37)
URAL = 26
SIBERIA = 27
YAKUTSK = 28
KAMCHATKA = 29
IRKUTSK = 30
MONGOLIA = 31
JAPAN = 32
AFGHANISTAN = 33
CHINA = 34
MIDDLE_EAST = 35
INDIA = 36
SIAM = 37

# Australia (ids 38-41)
INDONESIA = 38
NEW_GUINEA = 39
WESTERN_AUSTRALIA = 40
EASTERN_AUSTRALIA = 41


TERRITORIES: List[str] = [
    # North America
    "Alaska",                  # 0
    "Northwest Territory",     # 1
    "Greenland",               # 2
    "Alberta",                 # 3
    "Ontario",                 # 4
    "Quebec",                  # 5
    "Western United States",   # 6
    "Eastern United States",   # 7
    "Central America",         # 8
    # South America
    "Venezuela",               # 9
    "Peru",                    # 10
    "Brazil",                  # 11
    "Argentina",               # 12
    # Europe
    "Iceland",                 # 13
    "Scandinavia",             # 14
    "Great Britain",           # 15
    "Northern Europe",         # 16
    "Western Europe",          # 17
    "Southern Europe",         # 18
    "Ukraine",                 # 19
    # Africa
    "North Africa",            # 20
    "Egypt",                   # 21
    "East Africa",             # 22
    "Congo",                   # 23
    "South Africa",            # 24
    "Madagascar",              # 25
    # Asia
    "Ural",                    # 26
    "Siberia",                 # 27
    "Yakutsk",                 # 28
    "Kamchatka",               # 29
    "Irkutsk",                 # 30
    "Mongolia",                # 31
    "Japan",                   # 32
    "Afghanistan",             # 33
    "China",                   # 34
    "Middle East",             # 35
    "India",                   # 36
    "Siam",                    # 37
    # Australia
    "Indonesia",               # 38
    "New Guinea",              # 39
    "Western Australia",       # 40
    "Eastern Australia",       # 41
]


# ---------------------------------------------------------------------------
# Adjacency.  Authored as a list of undirected edges (each pair once), then
# expanded into the symmetric ``ADJ`` mapping below so the two directions can
# never drift apart.  Sea connections are grouped and commented explicitly.
# ---------------------------------------------------------------------------

_EDGES: List[Tuple[int, int]] = [
    # ---- North America (internal) ----
    (ALASKA, NORTHWEST_TERRITORY),
    (ALASKA, ALBERTA),
    (NORTHWEST_TERRITORY, ALBERTA),
    (NORTHWEST_TERRITORY, ONTARIO),
    (NORTHWEST_TERRITORY, GREENLAND),
    (GREENLAND, ONTARIO),
    (GREENLAND, QUEBEC),
    (ALBERTA, ONTARIO),
    (ALBERTA, WESTERN_UNITED_STATES),
    (ONTARIO, QUEBEC),
    (ONTARIO, WESTERN_UNITED_STATES),
    (ONTARIO, EASTERN_UNITED_STATES),
    (QUEBEC, EASTERN_UNITED_STATES),
    (WESTERN_UNITED_STATES, EASTERN_UNITED_STATES),
    (WESTERN_UNITED_STATES, CENTRAL_AMERICA),
    (EASTERN_UNITED_STATES, CENTRAL_AMERICA),

    # ---- South America (internal) ----
    (VENEZUELA, PERU),
    (VENEZUELA, BRAZIL),
    (PERU, BRAZIL),
    (PERU, ARGENTINA),
    (BRAZIL, ARGENTINA),

    # ---- Europe (internal) ----
    (ICELAND, SCANDINAVIA),
    (ICELAND, GREAT_BRITAIN),
    (SCANDINAVIA, GREAT_BRITAIN),
    (SCANDINAVIA, NORTHERN_EUROPE),
    (SCANDINAVIA, UKRAINE),
    (GREAT_BRITAIN, NORTHERN_EUROPE),
    (GREAT_BRITAIN, WESTERN_EUROPE),
    (NORTHERN_EUROPE, WESTERN_EUROPE),
    (NORTHERN_EUROPE, SOUTHERN_EUROPE),
    (NORTHERN_EUROPE, UKRAINE),
    (WESTERN_EUROPE, SOUTHERN_EUROPE),
    (SOUTHERN_EUROPE, UKRAINE),

    # ---- Africa (internal) ----
    (NORTH_AFRICA, EGYPT),
    (NORTH_AFRICA, EAST_AFRICA),
    (NORTH_AFRICA, CONGO),
    (EGYPT, EAST_AFRICA),
    (EAST_AFRICA, CONGO),
    (EAST_AFRICA, SOUTH_AFRICA),
    (EAST_AFRICA, MADAGASCAR),
    (CONGO, SOUTH_AFRICA),
    (SOUTH_AFRICA, MADAGASCAR),

    # ---- Asia (internal) ----
    (URAL, SIBERIA),
    (URAL, CHINA),
    (URAL, AFGHANISTAN),
    (SIBERIA, YAKUTSK),
    (SIBERIA, IRKUTSK),
    (SIBERIA, MONGOLIA),
    (SIBERIA, CHINA),
    (YAKUTSK, KAMCHATKA),
    (YAKUTSK, IRKUTSK),
    (KAMCHATKA, IRKUTSK),
    (KAMCHATKA, MONGOLIA),
    (KAMCHATKA, JAPAN),
    (IRKUTSK, MONGOLIA),
    (MONGOLIA, JAPAN),
    (MONGOLIA, CHINA),
    (AFGHANISTAN, CHINA),
    (AFGHANISTAN, MIDDLE_EAST),
    (AFGHANISTAN, INDIA),
    (CHINA, INDIA),
    (CHINA, SIAM),
    (MIDDLE_EAST, INDIA),
    (INDIA, SIAM),

    # ---- Australia (internal) ----
    (INDONESIA, NEW_GUINEA),
    (INDONESIA, WESTERN_AUSTRALIA),
    (NEW_GUINEA, WESTERN_AUSTRALIA),
    (NEW_GUINEA, EASTERN_AUSTRALIA),
    (WESTERN_AUSTRALIA, EASTERN_AUSTRALIA),

    # ---- Inter-continental SEA / bridge connections ----
    # North America <-> Asia
    (ALASKA, KAMCHATKA),
    # North America <-> Europe
    (GREENLAND, ICELAND),
    # North America <-> South America
    (CENTRAL_AMERICA, VENEZUELA),
    # South America <-> Africa
    (BRAZIL, NORTH_AFRICA),
    # Europe <-> Africa
    (WESTERN_EUROPE, NORTH_AFRICA),
    (SOUTHERN_EUROPE, NORTH_AFRICA),
    (SOUTHERN_EUROPE, EGYPT),
    # Europe <-> Asia
    (UKRAINE, URAL),
    (UKRAINE, AFGHANISTAN),
    (SOUTHERN_EUROPE, MIDDLE_EAST),
    (UKRAINE, MIDDLE_EAST),
    # Africa <-> Asia
    (EGYPT, MIDDLE_EAST),
    (EAST_AFRICA, MIDDLE_EAST),
    # Asia <-> Australia
    (SIAM, INDONESIA),
]


def _build_adj(edges: List[Tuple[int, int]], n: int) -> Dict[int, FrozenSet[int]]:
    """Expand an undirected edge list into a symmetric adjacency mapping."""
    acc: Dict[int, set] = {i: set() for i in range(n)}
    for a, b in edges:
        if a == b:
            raise ValueError(f"self-loop edge on territory {a}")
        acc[a].add(b)
        acc[b].add(a)
    return {i: frozenset(neighbors) for i, neighbors in acc.items()}


ADJ: Dict[int, FrozenSet[int]] = _build_adj(_EDGES, len(TERRITORIES))


# ---------------------------------------------------------------------------
# Continents -> (member territory ids, reinforcement bonus).
# ---------------------------------------------------------------------------

CONTINENTS: Dict[str, Tuple[FrozenSet[int], int]] = {
    "North America": (
        frozenset({
            ALASKA, NORTHWEST_TERRITORY, GREENLAND, ALBERTA, ONTARIO, QUEBEC,
            WESTERN_UNITED_STATES, EASTERN_UNITED_STATES, CENTRAL_AMERICA,
        }),
        5,
    ),
    "South America": (
        frozenset({VENEZUELA, PERU, BRAZIL, ARGENTINA}),
        2,
    ),
    "Europe": (
        frozenset({
            ICELAND, SCANDINAVIA, GREAT_BRITAIN, NORTHERN_EUROPE,
            WESTERN_EUROPE, SOUTHERN_EUROPE, UKRAINE,
        }),
        5,
    ),
    "Africa": (
        frozenset({
            NORTH_AFRICA, EGYPT, EAST_AFRICA, CONGO, SOUTH_AFRICA, MADAGASCAR,
        }),
        3,
    ),
    "Asia": (
        frozenset({
            URAL, SIBERIA, YAKUTSK, KAMCHATKA, IRKUTSK, MONGOLIA, JAPAN,
            AFGHANISTAN, CHINA, MIDDLE_EAST, INDIA, SIAM,
        }),
        7,
    ),
    "Australia": (
        frozenset({INDONESIA, NEW_GUINEA, WESTERN_AUSTRALIA, EASTERN_AUSTRALIA}),
        2,
    ),
}


# ---------------------------------------------------------------------------
# Risk-card set trade-in values.
# ---------------------------------------------------------------------------

SET_VALUES: List[int] = [4, 6, 8, 10, 12, 15]
SET_INCREMENT: int = 5


def set_value(n_sets_traded: int) -> int:
    """Value (armies) of the ``n_sets_traded``-th set turned in (0-indexed).

    The first six sets are worth ``SET_VALUES``; every set after that is worth
    the previous value + ``SET_INCREMENT`` (20, 25, 30, ...).
    """
    if n_sets_traded < 0:
        raise ValueError("n_sets_traded must be >= 0")
    if n_sets_traded < len(SET_VALUES):
        return SET_VALUES[n_sets_traded]
    extra = n_sets_traded - (len(SET_VALUES) - 1)
    return SET_VALUES[-1] + SET_INCREMENT * extra


__all__ = [
    "TERRITORIES",
    "ADJ",
    "CONTINENTS",
    "SET_VALUES",
    "SET_INCREMENT",
    "set_value",
]
