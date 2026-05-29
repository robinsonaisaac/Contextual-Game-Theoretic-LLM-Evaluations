"""Invariant tests for the standard 7-power Diplomacy map data table.

These tests guard the most error-prone Unit-S deliverable: the standard-map
adjacency/SC/home-centre data. They do NOT import any game logic, only the pure
data module `game_theory_llm.play.maps.diplomacy_map`.
"""

from __future__ import annotations

from game_theory_llm.play.maps import diplomacy_map as dm


# ---------------------------------------------------------------------------
# Structural basics
# ---------------------------------------------------------------------------

def test_province_ids_contiguous_and_unique():
    assert len(dm.PROVINCES) == len(set(dm.PROVINCES)), "duplicate province abbreviations"
    # ~75 provinces per spec
    assert 70 <= len(dm.PROVINCES) <= 80, f"unexpected province count {len(dm.PROVINCES)}"
    # KIND covers every id exactly
    assert set(dm.KIND) == set(range(len(dm.PROVINCES)))
    assert set(dm.KIND.values()) <= {"inland", "coastal", "sea"}


def test_exactly_34_supply_centres():
    assert len(dm.SUPPLY_CENTRE) == 34
    assert all(0 <= sc < len(dm.PROVINCES) for sc in dm.SUPPLY_CENTRE)
    # No SC sits on a sea province.
    assert all(dm.KIND[sc] != "sea" for sc in dm.SUPPLY_CENTRE)


def test_seven_powers_with_home_centres():
    expected = {"Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"}
    assert set(dm.HOME_SC) == expected
    # Russia has 4 home centres; everyone else has 3.
    assert len(dm.HOME_SC["Russia"]) == 4
    for power, homes in dm.HOME_SC.items():
        if power != "Russia":
            assert len(homes) == 3, f"{power} should have 3 home SCs"


def test_every_home_sc_is_a_supply_centre():
    for power, homes in dm.HOME_SC.items():
        for h in homes:
            assert h in dm.SUPPLY_CENTRE, f"home SC {dm.PROVINCES[h]} of {power} not in SUPPLY_CENTRE"


def test_home_centres_are_disjoint_across_powers():
    seen = set()
    for power, homes in dm.HOME_SC.items():
        overlap = seen & homes
        assert not overlap, f"home SCs shared across powers: {overlap}"
        seen |= homes


# ---------------------------------------------------------------------------
# Adjacency well-formedness
# ---------------------------------------------------------------------------

def test_adj_keys_cover_all_provinces():
    ids = set(range(len(dm.PROVINCES)))
    assert set(dm.ADJ_ARMY) == ids
    assert set(dm.ADJ_FLEET) == ids


def _assert_symmetric(adj):
    for a, neigh in adj.items():
        for b in neigh:
            assert a != b, f"self-loop at {dm.PROVINCES[a]}"
            assert a in adj[b], (
                f"asymmetric edge: {dm.PROVINCES[a]} -> {dm.PROVINCES[b]} "
                f"but not back"
            )


def test_army_adjacency_symmetric():
    _assert_symmetric(dm.ADJ_ARMY)


def test_fleet_adjacency_symmetric():
    _assert_symmetric(dm.ADJ_FLEET)


def test_army_adjacency_indices_in_range():
    n = len(dm.PROVINCES)
    for a, neigh in dm.ADJ_ARMY.items():
        for b in neigh:
            assert 0 <= b < n


def test_fleet_adjacency_indices_in_range():
    n = len(dm.PROVINCES)
    for a, neigh in dm.ADJ_FLEET.items():
        for b in neigh:
            assert 0 <= b < n


# ---------------------------------------------------------------------------
# Type-legality of moves
# ---------------------------------------------------------------------------

def test_armies_never_adjacent_into_a_sea():
    """An army may never move into a sea province, so no army edge may
    touch a sea on either endpoint."""
    for a, neigh in dm.ADJ_ARMY.items():
        if dm.KIND[a] == "sea":
            assert not neigh, f"sea province {dm.PROVINCES[a]} has army edges"
        for b in neigh:
            assert dm.KIND[b] != "sea", (
                f"army edge enters sea: {dm.PROVINCES[a]} -> {dm.PROVINCES[b]}"
            )


def test_fleets_never_on_inland_provinces():
    """Fleets occupy only coastal/sea; an inland province must have no
    fleet edges (neither as source nor as destination)."""
    for a, neigh in dm.ADJ_FLEET.items():
        if dm.KIND[a] == "inland":
            assert not neigh, f"inland province {dm.PROVINCES[a]} has fleet edges"
        for b in neigh:
            assert dm.KIND[b] != "inland", (
                f"fleet edge touches inland: {dm.PROVINCES[a]} -> {dm.PROVINCES[b]}"
            )


def test_every_coastal_province_has_at_least_one_fleet_edge():
    for a in range(len(dm.PROVINCES)):
        if dm.KIND[a] == "coastal":
            assert dm.ADJ_FLEET[a], (
                f"coastal province {dm.PROVINCES[a]} has no fleet adjacency"
            )


def test_every_sea_has_fleet_edges_and_no_army_edges():
    for a in range(len(dm.PROVINCES)):
        if dm.KIND[a] == "sea":
            assert dm.ADJ_FLEET[a], f"sea {dm.PROVINCES[a]} isolated for fleets"
            assert not dm.ADJ_ARMY[a], f"sea {dm.PROVINCES[a]} has army edges"


# ---------------------------------------------------------------------------
# Split coasts simplification (spec §9)
# ---------------------------------------------------------------------------

def test_split_coasts_modelled_as_single_provinces():
    # Spain / St Petersburg / Bulgaria each appear exactly once as a single
    # coastal province with no coast qualifiers.
    for abbr in ("SPA", "STP", "BUL"):
        assert abbr in dm.PROVINCES
        assert dm.KIND[dm.pid(abbr)] == "coastal"
    # COASTS is empty: the simplification means no split-coast tracking.
    assert dm.COASTS == {}


# ---------------------------------------------------------------------------
# Start units
# ---------------------------------------------------------------------------

def test_start_units_well_formed_and_on_home_centres():
    n = len(dm.PROVINCES)
    assert set(dm.START_UNITS) == set(dm.HOME_SC)
    for power, units in dm.START_UNITS.items():
        homes = dm.HOME_SC[power]
        for utype, prov, coast in units:
            assert utype in ("A", "F"), f"bad unit type {utype} for {power}"
            assert 0 <= prov < n, f"bad province id {prov} for {power}"
            assert coast is None, "split coasts are single provinces; coast must be None"
            assert prov in homes, (
                f"{power} starts a unit on {dm.PROVINCES[prov]} which is not a home SC"
            )
            # Type legality: fleets never start inland; armies never start on sea.
            if utype == "F":
                assert dm.KIND[prov] != "inland", (
                    f"{power} starts a fleet on inland {dm.PROVINCES[prov]}"
                )
            else:
                assert dm.KIND[prov] != "sea", (
                    f"{power} starts an army on sea {dm.PROVINCES[prov]}"
                )


def test_start_unit_counts_match_home_centres():
    # Standard opening: one unit per home centre.
    for power, units in dm.START_UNITS.items():
        assert len(units) == len(dm.HOME_SC[power]), (
            f"{power} unit count {len(units)} != home SC count {len(dm.HOME_SC[power])}"
        )


# ---------------------------------------------------------------------------
# Connectivity of the whole province graph (army ∪ fleet edges)
# ---------------------------------------------------------------------------

def test_province_graph_connected():
    """The union of army and fleet adjacency must connect all provinces
    (no orphan/island unreachable by any unit type)."""
    n = len(dm.PROVINCES)
    union = {i: set(dm.ADJ_ARMY[i]) | set(dm.ADJ_FLEET[i]) for i in range(n)}
    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(union[cur] - seen)
    missing = set(range(n)) - seen
    assert not missing, (
        "disconnected provinces: " + ", ".join(sorted(dm.PROVINCES[i] for i in missing))
    )


def test_no_isolated_province():
    """Every province must be reachable by at least one unit type."""
    for i in range(len(dm.PROVINCES)):
        assert dm.ADJ_ARMY[i] or dm.ADJ_FLEET[i], (
            f"province {dm.PROVINCES[i]} has no adjacency at all"
        )
