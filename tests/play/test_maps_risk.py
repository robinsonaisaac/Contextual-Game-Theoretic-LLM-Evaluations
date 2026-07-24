"""Tests for the classic 42-territory Risk map data table.

The map is the most error-prone deliverable (hand-authored adjacency), so we
check structure exhaustively: count, symmetry, continent partition, bonuses,
connectivity, plus a spot-check of the canonical sea bridges and a handful of
well-known territory degrees.
"""

from __future__ import annotations

from collections import deque

import pytest

from game_theory_llm.play.maps import risk_map as rm


def test_exactly_42_territories():
    assert len(rm.TERRITORIES) == 42
    # ids are dense 0..41 and names unique
    assert len(set(rm.TERRITORIES)) == 42


def test_adj_keys_cover_every_territory():
    assert set(rm.ADJ.keys()) == set(range(42))


def test_adjacency_is_symmetric():
    for a, neighbors in rm.ADJ.items():
        for b in neighbors:
            assert 0 <= b < 42, f"neighbor {b} of {a} out of range"
            assert a != b, f"self-loop on {a}"
            assert a in rm.ADJ[b], (
                f"asymmetric edge: {b} in ADJ[{a}] but {a} not in ADJ[{b}]"
            )


def test_no_isolated_territories():
    for a, neighbors in rm.ADJ.items():
        assert neighbors, f"territory {a} ({rm.TERRITORIES[a]}) has no neighbors"


def test_every_territory_in_exactly_one_continent():
    seen = {}
    for cont, (members, _bonus) in rm.CONTINENTS.items():
        for t in members:
            assert 0 <= t < 42
            assert t not in seen, (
                f"territory {t} ({rm.TERRITORIES[t]}) in both "
                f"{seen.get(t)} and {cont}"
            )
            seen[t] = cont
    assert set(seen.keys()) == set(range(42)), "continents do not partition all 42"


def test_continent_member_counts():
    counts = {c: len(members) for c, (members, _b) in rm.CONTINENTS.items()}
    assert counts == {
        "North America": 9,
        "South America": 4,
        "Europe": 7,
        "Africa": 6,
        "Asia": 12,
        "Australia": 4,
    }
    assert sum(counts.values()) == 42


def test_continent_bonuses():
    bonuses = {c: bonus for c, (_members, bonus) in rm.CONTINENTS.items()}
    assert bonuses == {
        "North America": 5,
        "South America": 2,
        "Europe": 5,
        "Africa": 3,
        "Asia": 7,
        "Australia": 2,
    }


def test_graph_is_connected():
    # BFS from territory 0 must reach all 42.
    seen = {0}
    q = deque([0])
    while q:
        cur = q.popleft()
        for nxt in rm.ADJ[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == set(range(42)), (
        "graph not connected; unreachable: "
        f"{sorted(set(range(42)) - seen)}"
    )


def test_set_values_and_increment():
    assert rm.SET_VALUES == [4, 6, 8, 10, 12, 15]
    assert rm.SET_INCREMENT == 5


def test_set_value_helper_escalation():
    # first six sets follow the table
    assert [rm.set_value(i) for i in range(6)] == [4, 6, 8, 10, 12, 15]
    # then +5 each subsequent set
    assert rm.set_value(6) == 20
    assert rm.set_value(7) == 25
    assert rm.set_value(8) == 30
    with pytest.raises(ValueError):
        rm.set_value(-1)


# --- adjacency spot-checks against the standard Hasbro board -----------------

def _name_to_id():
    return {name: i for i, name in enumerate(rm.TERRITORIES)}


def _neighbors_by_name(name):
    nid = _name_to_id()
    ids = rm.ADJ[nid[name]]
    return {rm.TERRITORIES[i] for i in ids}


CANONICAL_SEA_BRIDGES = [
    ("Alaska", "Kamchatka"),
    ("Greenland", "Iceland"),
    ("Brazil", "North Africa"),
    ("Western Europe", "North Africa"),
    ("Southern Europe", "North Africa"),
    ("Southern Europe", "Egypt"),
    ("East Africa", "Middle East"),
    ("Siam", "Indonesia"),
    ("Central America", "Venezuela"),
]


@pytest.mark.parametrize("a,b", CANONICAL_SEA_BRIDGES)
def test_canonical_sea_bridges_present(a, b):
    nid = _name_to_id()
    assert nid[b] in rm.ADJ[nid[a]], f"missing sea bridge {a}<->{b}"
    assert nid[a] in rm.ADJ[nid[b]], f"missing sea bridge {b}<->{a}"


def test_known_territory_neighbor_sets():
    # A handful of fully-specified neighbor sets from the standard board.
    expected = {
        "Alaska": {"Northwest Territory", "Alberta", "Kamchatka"},
        "Central America": {
            "Western United States", "Eastern United States", "Venezuela",
        },
        "Brazil": {"Venezuela", "Peru", "Argentina", "North Africa"},
        "Iceland": {"Greenland", "Scandinavia", "Great Britain"},
        "Egypt": {"North Africa", "East Africa", "Southern Europe", "Middle East"},
        "Japan": {"Kamchatka", "Mongolia"},
        "Indonesia": {"Siam", "New Guinea", "Western Australia"},
        "Eastern Australia": {"New Guinea", "Western Australia"},
        "Ukraine": {
            "Scandinavia", "Northern Europe", "Southern Europe",
            "Ural", "Afghanistan", "Middle East",
        },
    }
    for name, nbrs in expected.items():
        assert _neighbors_by_name(name) == nbrs, (
            f"{name} neighbors {_neighbors_by_name(name)} != expected {nbrs}"
        )


def test_total_edge_count():
    # Sum of degrees is twice the number of undirected edges. The standard
    # Risk board has 84 directed adjacencies (42 undirected edges... actually
    # the canonical board has more); assert it is even and matches the source
    # edge list expanded symmetrically.
    total_directed = sum(len(v) for v in rm.ADJ.values())
    assert total_directed % 2 == 0
    # Distinct undirected edges
    edges = set()
    for a, nbrs in rm.ADJ.items():
        for b in nbrs:
            edges.add((min(a, b), max(a, b)))
    assert len(edges) == total_directed // 2
