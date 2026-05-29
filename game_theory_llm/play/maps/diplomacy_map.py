"""Standard 7-power Europe map data for Diplomacy (Unit-S data table).

This module is PURE DATA — no game logic. Unit 4 (`diplomacy_lite.py`) consumes
the symbols `PROVINCES`, `KIND`, `SUPPLY_CENTRE`, `HOME_SC`, `ADJ_ARMY`,
`ADJ_FLEET`, `COASTS`, and `START_UNITS` to drive the full-press adjudicator.

Province identity
-----------------
Provinces are referenced everywhere by their **integer id**, which is the index
into `PROVINCES`. `PROVINCES[i]` is the canonical 3-letter abbreviation. There
are 75 provinces (19 sea, the rest land split into inland/coastal), and EXACTLY
34 supply centres, matching the standard map.

Adjacency model
----------------
Two distinct graphs are provided because Diplomacy units move differently:

* `ADJ_ARMY[i]` — provinces an **army** on province ``i`` may move to. Armies
  travel over land only; they may never enter a ``"sea"`` province, and there
  is never a land edge into a sea province (enforced by tests). Coastal land
  provinces that are land-adjacent (e.g. Gascony–Spain) are connected here.

* `ADJ_FLEET[i]` — provinces a **fleet** on province ``i`` may move to. Fleets
  occupy only ``"coastal"`` and ``"sea"`` provinces. Fleet edges run sea↔sea,
  sea↔coastal (where the coast actually touches that body of water), and
  coastal↔coastal where two coasts are adjacent along a shared shoreline (e.g.
  Brest–Gascony, Marseilles–Spain). Two coastal provinces that share only an
  inland border (no common water) are NOT fleet-adjacent even though they are
  army-adjacent.

Both graphs are SYMMETRIC by construction: every edge is declared once in the
edge lists below and inserted in both directions by `_build_adj`.

Simplifications (spec §5.4 / §9 — `config.adjudicator = "lite-press-v2"`)
-------------------------------------------------------------------------
* **Split coasts modelled as single provinces.** Spain (SPA), St Petersburg
  (STP), and Bulgaria (BUL) have, on the real board, two distinct named coasts
  (north/south, etc.) that a fleet must pick between. Here each is a SINGLE
  province: a fleet entering Spain simply occupies ``SPA`` with no coast
  qualifier, and `ADJ_FLEET[SPA]` is the UNION of both real coasts' sea
  neighbours. `COASTS` is therefore empty (``{}``) — there are no split-coast
  qualifiers to track. This makes a handful of fleet moves legal that the full
  rules would forbid (e.g. a fleet may "pass through" Spain between MAO and WES
  via two moves), which we accept as a documented lite-press deviation.
* The remaining §9 simplifications (single-fleet convoys, Szykman paradox
  handling, deterministic retreats/builds) are adjudicator concerns and live in
  Unit 4, not in this data table.

START_UNITS
-----------
`START_UNITS[power]` is the standard 1901 opening: a list of
``(unit_type, province_id, coast)`` triples where ``unit_type`` is ``"A"`` or
``"F"`` and ``coast`` is always ``None`` (split coasts are single provinces, so
no opening unit needs a coast qualifier — notably StP's fleet just sits on STP).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Provinces (id == index). 75 total.
# ---------------------------------------------------------------------------
# Grouped by region for readability; the flat list defines the ids.

PROVINCES: List[str] = [
    # Seas (19): ids 0..18
    "ADR", "AEG", "BAL", "BAR", "BLA", "BOT", "EAS", "ENG", "GOL", "HEL",
    "ION", "IRI", "MAO", "NAO", "NTH", "NWG", "SKA", "TYS", "WES",
    # England: 19..24
    "CLY", "EDI", "LVP", "YOR", "WAL", "LON",
    # France/Iberia: 25..32
    "BRE", "PIC", "PAR", "BUR", "GAS", "MAR", "SPA", "POR",
    # NAfrica/Low/Germany: 33..42
    "NAF", "TUN", "BEL", "HOL", "RUH", "KIE", "MUN", "BER", "PRU", "SIL",
    # Italy: 43..48
    "PIE", "TUS", "ROM", "VEN", "APU", "NAP",
    # Austria/Balkans: 49..59
    "TYR", "BOH", "VIE", "TRI", "BUD", "GAL", "SER", "ALB", "GRE", "BUL", "RUM",
    # Russia/Scandinavia: 60..69
    "STP", "MOS", "WAR", "LVN", "UKR", "SEV", "FIN", "NWY", "SWE", "DEN",
    # Turkey/Near East: 70..74
    "CON", "ANK", "SMY", "ARM", "SYR",
]

# id lookup by abbreviation
_ID: Dict[str, int] = {abbr: i for i, abbr in enumerate(PROVINCES)}


def pid(abbr: str) -> int:
    """Return the integer id for a province abbreviation (helper for callers)."""
    return _ID[abbr]


# ---------------------------------------------------------------------------
# Province kind: "inland" | "coastal" | "sea"
# ---------------------------------------------------------------------------

_SEAS = {
    "ADR", "AEG", "BAL", "BAR", "BLA", "BOT", "EAS", "ENG", "GOL", "HEL",
    "ION", "IRI", "MAO", "NAO", "NTH", "NWG", "SKA", "TYS", "WES",
}

# Land provinces with NO sea coast (armies only; fleets may never occupy).
_INLAND = {
    "PAR", "BUR", "RUH", "MUN", "SIL",
    "TYR", "BOH", "VIE", "BUD", "GAL", "SER",
    "MOS", "WAR", "UKR",
}

KIND: Dict[int, str] = {}
for _abbr in PROVINCES:
    if _abbr in _SEAS:
        KIND[_ID[_abbr]] = "sea"
    elif _abbr in _INLAND:
        KIND[_ID[_abbr]] = "inland"
    else:
        KIND[_ID[_abbr]] = "coastal"


# ---------------------------------------------------------------------------
# Supply centres (EXACTLY 34) and home centres.
# ---------------------------------------------------------------------------

_SC_ABBR = [
    # Austria (3 home) + Balkans
    "VIE", "BUD", "TRI",
    "SER", "GRE", "RUM", "BUL",
    # England (3 home)
    "EDI", "LVP", "LON",
    # France (3 home) + Iberia
    "BRE", "PAR", "MAR",
    "SPA", "POR",
    # Germany (3 home) + Low Countries
    "KIE", "BER", "MUN",
    "BEL", "HOL", "DEN",
    # Italy (3 home) + Tunis
    "VEN", "ROM", "NAP",
    "TUN",
    # Russia (4 home) + Scandinavia
    "MOS", "STP", "WAR", "SEV",
    "NWY", "SWE",
    # Turkey (3 home) + Smyrna already counted; near-east
    "CON", "ANK", "SMY",
]

SUPPLY_CENTRE: FrozenSet[int] = frozenset(_ID[a] for a in _SC_ABBR)

HOME_SC: Dict[str, FrozenSet[int]] = {
    "Austria": frozenset(_ID[a] for a in ("VIE", "BUD", "TRI")),
    "England": frozenset(_ID[a] for a in ("EDI", "LVP", "LON")),
    "France": frozenset(_ID[a] for a in ("BRE", "PAR", "MAR")),
    "Germany": frozenset(_ID[a] for a in ("KIE", "BER", "MUN")),
    "Italy": frozenset(_ID[a] for a in ("VEN", "ROM", "NAP")),
    "Russia": frozenset(_ID[a] for a in ("MOS", "STP", "WAR", "SEV")),
    "Turkey": frozenset(_ID[a] for a in ("CON", "ANK", "SMY")),
}


# ---------------------------------------------------------------------------
# Adjacency edge lists. Each edge declared ONCE; `_build_adj` symmetrises.
# ---------------------------------------------------------------------------

def _build_adj(edges: List[Tuple[str, str]]) -> Dict[int, FrozenSet[int]]:
    """Turn an undirected edge list of abbreviations into a symmetric
    id->frozenset adjacency map. Every province id appears as a key."""
    tmp: Dict[int, Set[int]] = {i: set() for i in range(len(PROVINCES))}
    for a, b in edges:
        ia, ib = _ID[a], _ID[b]
        tmp[ia].add(ib)
        tmp[ib].add(ia)
    return {i: frozenset(ns) for i, ns in tmp.items()}


# ----- ARMY edges (land borders only; never touch a sea) -------------------
# Coasts that are land-adjacent are connected here; seas never appear.
_ARMY_EDGES: List[Tuple[str, str]] = [
    # England (island; armies move only within Britain — fleets handle the seas)
    ("CLY", "EDI"), ("CLY", "LVP"),
    ("EDI", "YOR"), ("EDI", "LVP"),
    ("LVP", "YOR"), ("LVP", "WAL"),
    ("YOR", "WAL"), ("YOR", "LON"),
    ("WAL", "LON"),

    # France / Iberia
    ("BRE", "PIC"), ("BRE", "PAR"), ("BRE", "GAS"),
    ("PIC", "PAR"), ("PIC", "BUR"), ("PIC", "BEL"),
    ("PAR", "BUR"), ("PAR", "GAS"),
    ("BUR", "GAS"), ("BUR", "MAR"), ("BUR", "BEL"), ("BUR", "RUH"), ("BUR", "MUN"),
    ("GAS", "MAR"), ("GAS", "SPA"),
    ("MAR", "SPA"), ("MAR", "PIE"),
    ("SPA", "POR"),

    # North Africa / Tunis
    ("NAF", "TUN"),

    # Low Countries / Germany
    ("BEL", "HOL"), ("BEL", "RUH"),
    ("HOL", "RUH"), ("HOL", "KIE"),
    ("RUH", "KIE"), ("RUH", "MUN"),
    ("KIE", "MUN"), ("KIE", "BER"), ("KIE", "DEN"),
    ("MUN", "BER"), ("MUN", "SIL"), ("MUN", "BOH"), ("MUN", "TYR"),
    ("BER", "PRU"), ("BER", "SIL"),
    ("PRU", "SIL"), ("PRU", "WAR"), ("PRU", "LVN"),
    ("SIL", "BOH"), ("SIL", "GAL"), ("SIL", "WAR"),

    # Italy
    ("PIE", "TUS"), ("PIE", "VEN"), ("PIE", "TYR"),
    ("TUS", "ROM"), ("TUS", "VEN"),
    ("ROM", "VEN"), ("ROM", "NAP"), ("ROM", "APU"),
    ("VEN", "TYR"), ("VEN", "TRI"), ("VEN", "APU"),
    ("APU", "NAP"),

    # Austria / Balkans
    ("TYR", "BOH"), ("TYR", "VIE"), ("TYR", "TRI"), ("TYR", "MUN"),
    ("BOH", "VIE"), ("BOH", "GAL"),
    ("VIE", "TRI"), ("VIE", "BUD"), ("VIE", "GAL"),
    ("TRI", "BUD"), ("TRI", "SER"), ("TRI", "ALB"),
    ("BUD", "GAL"), ("BUD", "RUM"), ("BUD", "SER"),
    ("GAL", "RUM"), ("GAL", "WAR"), ("GAL", "UKR"),
    ("SER", "ALB"), ("SER", "GRE"), ("SER", "BUL"), ("SER", "RUM"),
    ("ALB", "GRE"),
    ("GRE", "BUL"),
    ("BUL", "RUM"), ("BUL", "CON"),
    ("RUM", "SEV"), ("RUM", "UKR"),

    # Russia / Scandinavia
    ("STP", "MOS"), ("STP", "FIN"), ("STP", "LVN"), ("STP", "NWY"),
    ("MOS", "WAR"), ("MOS", "LVN"), ("MOS", "UKR"), ("MOS", "SEV"),
    ("WAR", "LVN"), ("WAR", "UKR"),
    ("LVN", "PRU"),
    ("UKR", "SEV"),
    ("FIN", "NWY"), ("FIN", "SWE"),
    ("NWY", "SWE"),
    ("SWE", "DEN"),

    # Turkey / Near East
    ("CON", "ANK"), ("CON", "SMY"),
    ("ANK", "SMY"), ("ANK", "ARM"),
    ("SMY", "ARM"), ("SMY", "SYR"),
    ("ARM", "SYR"), ("ARM", "SEV"),
]

ADJ_ARMY: Dict[int, FrozenSet[int]] = _build_adj(_ARMY_EDGES)


# ----- FLEET edges (sea<->sea, sea<->coast, coast<->coast along water) ------
_FLEET_EDGES: List[Tuple[str, str]] = [
    # ---- Sea-to-sea ----
    ("NAO", "NWG"), ("NAO", "IRI"), ("NAO", "MAO"),
    ("NWG", "BAR"), ("NWG", "NTH"),
    ("BAR", "STP"),  # also coast below; declared once
    ("NTH", "SKA"), ("NTH", "HEL"), ("NTH", "ENG"),
    ("SKA", "BAL"),
    ("BAL", "BOT"),
    ("HEL", "ENG"),
    ("IRI", "ENG"), ("IRI", "MAO"),
    ("ENG", "MAO"),
    ("MAO", "WES"),
    ("WES", "GOL"), ("WES", "TYS"),
    ("GOL", "TYS"),
    ("TYS", "ION"), ("TYS", "ADR"),
    ("ION", "ADR"), ("ION", "AEG"), ("ION", "EAS"),
    ("AEG", "EAS"),

    # ---- North Atlantic / British Isles coasts ----
    ("NAO", "CLY"), ("NAO", "LVP"),
    ("NWG", "CLY"), ("NWG", "EDI"), ("NWG", "NWY"),
    ("IRI", "LVP"), ("IRI", "WAL"),
    ("CLY", "EDI"), ("CLY", "LVP"),
    ("EDI", "YOR"), ("EDI", "NTH"),
    ("LVP", "WAL"),
    ("YOR", "WAL"), ("YOR", "LON"), ("YOR", "NTH"),
    ("WAL", "LON"), ("WAL", "ENG"),
    ("LON", "NTH"), ("LON", "ENG"),

    # ---- North Sea / Channel / Low Countries ----
    ("NTH", "NWY"), ("NTH", "DEN"), ("NTH", "HOL"), ("NTH", "BEL"),
    ("ENG", "PIC"), ("ENG", "BRE"), ("ENG", "BEL"),
    ("HEL", "DEN"), ("HEL", "KIE"), ("HEL", "HOL"),
    ("BEL", "HOL"), ("BEL", "PIC"),
    ("HOL", "KIE"),

    # ---- Atlantic / Iberian / W-Med coasts ----
    ("MAO", "BRE"), ("MAO", "GAS"), ("MAO", "SPA"), ("MAO", "POR"), ("MAO", "NAF"),
    ("BRE", "PIC"), ("BRE", "GAS"),
    ("GAS", "SPA"),
    ("SPA", "POR"), ("SPA", "GOL"), ("SPA", "WES"), ("SPA", "MAR"),
    ("POR", "SPA"),
    ("WES", "NAF"), ("WES", "TUN"),
    ("NAF", "TUN"),
    ("GOL", "MAR"), ("GOL", "PIE"), ("GOL", "TUS"),
    ("MAR", "PIE"),

    # ---- Tyrrhenian / Italy / Ionian ----
    ("TYS", "TUS"), ("TYS", "ROM"), ("TYS", "NAP"), ("TYS", "TUN"),
    ("TUS", "ROM"), ("TUS", "PIE"),
    ("ROM", "NAP"),
    ("NAP", "APU"), ("NAP", "ION"),
    ("APU", "ION"), ("APU", "ADR"), ("APU", "VEN"),
    ("ION", "TUN"), ("ION", "GRE"), ("ION", "ALB"),

    # ---- Adriatic / Balkan W coast ----
    ("ADR", "VEN"), ("ADR", "TRI"), ("ADR", "ALB"),
    ("VEN", "TRI"),
    ("TRI", "ALB"),
    ("ALB", "GRE"),

    # ---- Aegean / E-Med / Turkey-S ----
    ("AEG", "GRE"), ("AEG", "BUL"), ("AEG", "CON"), ("AEG", "SMY"),
    ("EAS", "SMY"), ("EAS", "SYR"),
    ("SMY", "SYR"), ("SMY", "CON"),
    ("GRE", "BUL"),

    # ---- Black Sea ring ----
    ("BLA", "BUL"), ("BLA", "RUM"), ("BLA", "SEV"), ("BLA", "ARM"), ("BLA", "ANK"), ("BLA", "CON"),
    ("BUL", "CON"), ("BUL", "RUM"),
    ("RUM", "SEV"),
    ("SEV", "ARM"),
    ("ANK", "CON"), ("ANK", "ARM"),

    # ---- Baltic / Gulf of Bothnia / Scandinavia / Russia-N ----
    ("BAL", "DEN"), ("BAL", "KIE"), ("BAL", "BER"), ("BAL", "PRU"), ("BAL", "LVN"), ("BAL", "SWE"),
    ("BOT", "SWE"), ("BOT", "FIN"), ("BOT", "STP"), ("BOT", "LVN"),
    ("SKA", "NWY"), ("SKA", "SWE"), ("SKA", "DEN"),
    ("DEN", "KIE"), ("DEN", "SWE"),
    ("KIE", "BER"), ("KIE", "HOL"),
    ("BER", "PRU"),
    ("PRU", "LVN"),
    ("LVN", "STP"),
    ("SWE", "FIN"), ("SWE", "NWY"),
    ("FIN", "STP"),
    ("NWY", "STP"),
    ("BAR", "NWY"),
]

ADJ_FLEET: Dict[int, FrozenSet[int]] = _build_adj(_FLEET_EDGES)


# ---------------------------------------------------------------------------
# Split coasts: modelled as single provinces (spec §9). No coast qualifiers.
# ---------------------------------------------------------------------------
# On the real board, SPA / STP / BUL have two named coasts; here each is one
# province whose ADJ_FLEET is the union of both coasts' sea neighbours. We
# expose COASTS as an empty dict to signal "no split-coast modelling".
COASTS: Dict[int, List[str]] = {}


# ---------------------------------------------------------------------------
# Standard 1901 opening units. coast is always None (single-province coasts).
# ---------------------------------------------------------------------------
# Real opening: Austria F Tri, A Vie, A Bud; England F Edi, F Lon, A Lvp;
# France F Bre, A Par, A Mar; Germany F Kie, A Ber, A Mun;
# Italy F Nap, A Rom, A Ven; Russia A Mos, F StP, A War, F Sev;
# Turkey F Ank, A Con, A Smy.

START_UNITS: Dict[str, List[Tuple[str, int, Optional[str]]]] = {
    "Austria": [
        ("F", _ID["TRI"], None),
        ("A", _ID["VIE"], None),
        ("A", _ID["BUD"], None),
    ],
    "England": [
        ("F", _ID["EDI"], None),
        ("F", _ID["LON"], None),
        ("A", _ID["LVP"], None),
    ],
    "France": [
        ("F", _ID["BRE"], None),
        ("A", _ID["PAR"], None),
        ("A", _ID["MAR"], None),
    ],
    "Germany": [
        ("F", _ID["KIE"], None),
        ("A", _ID["BER"], None),
        ("A", _ID["MUN"], None),
    ],
    "Italy": [
        ("F", _ID["NAP"], None),
        ("A", _ID["ROM"], None),
        ("A", _ID["VEN"], None),
    ],
    "Russia": [
        ("A", _ID["MOS"], None),
        ("F", _ID["STP"], None),
        ("A", _ID["WAR"], None),
        ("F", _ID["SEV"], None),
    ],
    "Turkey": [
        ("F", _ID["ANK"], None),
        ("A", _ID["CON"], None),
        ("A", _ID["SMY"], None),
    ],
}


__all__ = [
    "PROVINCES",
    "KIND",
    "SUPPLY_CENTRE",
    "HOME_SC",
    "ADJ_ARMY",
    "ADJ_FLEET",
    "COASTS",
    "START_UNITS",
    "pid",
]
