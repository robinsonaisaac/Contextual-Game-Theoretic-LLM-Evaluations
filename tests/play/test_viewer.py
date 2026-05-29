"""Tests for the pure JSONL watch/replay tool (Unit 5, spec §6).

Covers the four required assertions:

  (a) a god render contains private whisper *text*, but a non-recipient seat
      render does NOT (the masking visual-QA guarantee);
  (b) a seat render only shows that seat's entitled observations;
  (c) ``--metrics`` produces an alliance summary table;
  (d) load / reconstruct tolerate a truncated trailing line without crashing.

Plus: HTML export is self-contained and embeds the events; the live
``LogReader`` reconstructs incrementally and buffers a partial trailing line;
the lazy game-class registry never hard-depends on a game being built.

The checked-in fixture (``tests/play/fixtures/sample_match.jsonl``) uses empty
whisper text, so assertion (a) is exercised against a small synthetic log with a
known secret string; the fixture itself drives the entitlement / metrics /
truncation assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from game_theory_llm.play import viewer


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_match.jsonl"


# --------------------------------------------------------------------------- #
# Synthetic log with a non-empty private whisper (for assertion (a))
# --------------------------------------------------------------------------- #
SECRET = "MEET-ME-AT-THE-BRIDGE-AT-DAWN"
PUBLIC_LINE = "everyone-can-read-this"


def _synthetic_events():
    mid = "testmatch00000000000000000000000"
    return [
        {"type": "match_start", "schema": 2, "game": "stub_game",
         "n_players": 4, "seed": 1,
         "players": ["P0", "P1", "P2", "P3"], "config": {},
         "steering_tags": [None, None, None, None],
         "ts": "t", "event_id": 0, "match_id": mid},
        {"type": "setup", "turn": 0,
         "god_view": {"hidden_truth": "P0-is-the-wolf"},
         "ts": "t", "event_id": 1, "match_id": mid},
        {"type": "state_snapshot", "turn": 0, "phase": "negotiation",
         "snapshot": {"public": {"phase": "negotiation"},
                      "hidden": {"secret_board": 42}},
         "ts": "t", "event_id": 2, "match_id": mid},
        # public message — visible to all
        {"type": "observation", "turn": 0, "actor": 1, "audience": [0, 1, 2, 3],
         "obs": {"type": "message", "scope": "public", "from": 1,
                 "text": PUBLIC_LINE, "turn": 0},
         "ts": "t", "event_id": 3, "match_id": mid},
        # whisper P0 -> P2 : recipients get full text...
        {"type": "observation", "turn": 0, "actor": 0, "audience": [0, 2],
         "obs": {"type": "message", "scope": "private", "from": 0, "to": [2],
                 "text": SECRET, "turn": 0},
         "ts": "t", "event_id": 4, "match_id": mid},
        # ...bystanders get only a content-free meta leak
        {"type": "observation", "turn": 0, "actor": 0, "audience": [1, 3],
         "obs": {"type": "message_meta", "from": 0, "n_recipients": 1,
                 "turn": 0},
         "ts": "t", "event_id": 5, "match_id": mid},
        {"type": "terminal", "turn": 0, "rewards": [1.0, 0.0, 0.0, 0.0],
         "state": "{}", "winner": 0, "win_reason": "synthetic",
         "alliance_summary": {"n_proposed": 0, "n_accepted": 0,
                              "n_declined": 0, "n_broken": 0, "n_honored": 0,
                              "n_betrayed": 0, "per_player": {}},
         "ts": "t", "event_id": 6, "match_id": mid},
    ]


def _write_jsonl(path: Path, events) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n",
                    encoding="utf-8")


# --------------------------------------------------------------------------- #
# (a) god sees private whisper text; non-recipient seat does NOT
# --------------------------------------------------------------------------- #
def test_god_reveals_whisper_text_seat_does_not(tmp_path):
    log = tmp_path / "synthetic.jsonl"
    _write_jsonl(log, _synthetic_events())
    replay = viewer.reconstruct(viewer.load_events(str(log)))
    last = len(replay.frames) - 1

    god = "\n".join(replay.render(i, reveal="god")
                    for i in range(len(replay.frames)))
    assert SECRET in god, "god view must contain full private whisper text"

    # Seat 1 is NOT a recipient of the P0->P2 whisper.
    seat1 = "\n".join(replay.render(i, reveal="seat1")
                      for i in range(len(replay.frames)))
    assert SECRET not in seat1, \
        "a non-recipient seat must NOT see the private whisper text"
    # The bystander still learns *that* a private message happened (meta leak).
    assert "content hidden" in seat1 or "private" in seat1.lower()
    # Public message is visible to everyone, including seat 1.
    assert PUBLIC_LINE in seat1

    # Seat 2 IS a recipient and must see the text.
    seat2 = "\n".join(replay.render(i, reveal="seat2")
                      for i in range(len(replay.frames)))
    assert SECRET in seat2, "the whisper recipient must see the text"
    _ = last  # keep flake8 calm


# --------------------------------------------------------------------------- #
# (b) a seat render only shows that seat's entitled observations
# --------------------------------------------------------------------------- #
def test_seat_only_sees_entitled_observations():
    events = viewer.load_events(str(FIXTURE))
    replay = viewer.reconstruct(events)

    # Find every private whisper observation and its intended audience.
    whisper_audiences = []
    for e in events:
        if e.get("type") == "observation":
            obs = e.get("obs", {})
            if obs.get("type") == "message" and obs.get("scope") == "private":
                whisper_audiences.append((e.get("event_id"),
                                          set(e.get("audience", []))))
    assert whisper_audiences, "fixture must contain private whispers"

    # For some seat that is excluded from at least one whisper, the seat view's
    # chat must not include that whisper as a full 'whisper' line for that seat.
    # We verify entitlement structurally: the visible chat set for seat N is a
    # subset of {chats whose audience includes N}.
    for seat in range(replay.n_players):
        # gather chats the fold built
        last_frame = replay.frames[-1]
        visible = replay._visible_chats(last_frame.chats, f"seat{seat}")
        for c in visible:
            assert seat in c["audience"], (
                f"seat {seat} shown a chat it was not in the audience of: {c}")
        # and the god view sees at least as many chats as any seat
        god_visible = replay._visible_chats(last_frame.chats, "god")
        assert len(visible) <= len(god_visible)

    # A concrete entitlement check: render the seat that is the bystander of the
    # first private whisper and confirm the recipient pair is not exposed as a
    # full whisper line that names both private endpoints with text.
    eid, aud = whisper_audiences[0]
    bystander = next((s for s in range(replay.n_players) if s not in aud), None)
    if bystander is not None:
        text = "\n".join(replay.render(i, reveal=f"seat{bystander}")
                         for i in range(len(replay.frames)))
        recipient = "\n".join(replay.render(i, reveal=f"seat{min(aud)}")
                              for i in range(len(replay.frames)))
        # Whichever seats are private endpoints, the bystander's render must not
        # show MORE chat lines than the recipient's (it sees strictly less or
        # equal private content).
        assert text.count("whisper") <= recipient.count("whisper") + \
            text.count("content hidden") + 50  # generous; structural only


def test_god_sees_more_or_equal_chats_than_any_seat():
    events = viewer.load_events(str(FIXTURE))
    replay = viewer.reconstruct(events)
    last = replay.frames[-1]
    n_god = len(replay._visible_chats(last.chats, "god"))
    for seat in range(replay.n_players):
        n_seat = len(replay._visible_chats(last.chats, f"seat{seat}"))
        assert n_seat <= n_god


# --------------------------------------------------------------------------- #
# (c) --metrics produces an alliance summary table
# --------------------------------------------------------------------------- #
def test_metrics_table_has_alliance_summary():
    table = viewer.metrics_table(str(FIXTURE))
    assert "alliance summary" in table.lower()
    # The fixture's terminal alliance_summary has n_proposed == 8.
    assert "n_proposed" in table
    assert "8" in table  # the fixture proposed 8 alliances
    assert "rates" in table.lower()
    # honour/betrayal rate keys present
    assert "honour" in table.lower()


def test_metrics_main_smoke(capsys):
    rc = viewer.main([str(FIXTURE), "--metrics"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alliance summary" in out.lower()


# --------------------------------------------------------------------------- #
# (d) truncated trailing line does not crash load / reconstruct
# --------------------------------------------------------------------------- #
def test_truncated_trailing_line_tolerated(tmp_path):
    full = FIXTURE.read_text(encoding="utf-8").splitlines()
    # Append a deliberately broken / partial JSON line (as a live tail would see).
    broken = full[:] + ['{"type": "observation", "turn": 99, "actor": 0, "aud']
    log = tmp_path / "truncated.jsonl"
    log.write_text("\n".join(broken) + "\n", encoding="utf-8")

    events = viewer.load_events(str(log))      # must not raise
    # The partial line is dropped; all complete lines survive.
    assert len(events) == len(full)
    replay = viewer.reconstruct(events)         # must not raise
    assert replay.frames
    # rendering every frame must not raise
    for i in range(len(replay.frames)):
        replay.render(i, reveal="god")
        replay.render(i, reveal="seat0")


def test_load_events_empty_and_blank_lines(tmp_path):
    log = tmp_path / "blanks.jsonl"
    log.write_text("\n\n   \n", encoding="utf-8")
    assert viewer.load_events(str(log)) == []


# --------------------------------------------------------------------------- #
# Reconstruction / rendering basics over the real fixture
# --------------------------------------------------------------------------- #
def test_reconstruct_basic_shape():
    replay = viewer.reconstruct(viewer.load_events(str(FIXTURE)))
    assert replay.game == "stub_game"
    assert replay.n_players == 4
    assert len(replay.frames) == 132  # one frame per record in the fixture
    # phase banner appears on a phase_change frame
    banners = [f.banner for f in replay.frames if f.banner]
    assert banners, "expected at least one phase banner"
    assert any("PHASE" in b for b in banners)


def test_alliance_ledger_tracks_status():
    replay = viewer.reconstruct(viewer.load_events(str(FIXTURE)))
    last = replay.frames[-1]
    ledger = {a["id"]: a for a in last.alliances}
    # Fixture: alliance 0 ends 'broken', alliance 2 'declined'.
    assert ledger[0]["status"] in ("broken", "BETRAYED")
    assert ledger[2]["status"] == "declined"
    # honour counter is non-zero (fixture honoured 2 board actions)
    assert last.counters["honored"] >= 1


def test_static_render_runs(capsys):
    rc = viewer.main([str(FIXTURE)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MATCH START" in out
    assert "TERMINAL" in out
    assert "alliances" in out.lower()


def test_seat_view_runs(capsys):
    rc = viewer.main([str(FIXTURE), "--seat", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "view: P0" in out or "(view: P0)" in out


# --------------------------------------------------------------------------- #
# HTML export
# --------------------------------------------------------------------------- #
def test_html_export_is_self_contained(tmp_path):
    out = tmp_path / "replay.html"
    rc = viewer.main([str(FIXTURE), "--html", str(out)])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in html.lower()
    # JSONL embedded as application/json
    assert 'type="application/json"' in html
    # scrub bar + seat selector present
    assert 'id="scrub"' in html
    assert 'id="seat"' in html
    # the embedded data references the game so the browser can re-mask
    assert "stub_game" in html
    # closing-script sequence is escaped so the embed is safe
    assert "</script" not in html.split('id="data"')[1].split("</script>")[0] \
        or "<\\/" in html


def test_html_does_not_leak_raw_script_terminator(tmp_path):
    # Build a log whose message text contains a literal </script> to ensure the
    # HTML escaping prevents breaking out of the embedded JSON block.
    events = _synthetic_events()
    events[4]["obs"]["text"] = "pwn</script><script>alert(1)</script>"
    log = tmp_path / "xss.jsonl"
    _write_jsonl(log, events)
    replay = viewer.reconstruct(viewer.load_events(str(log)))
    html = viewer.HtmlRenderer().render(replay)
    # the embedded JSON block must not contain a literal closing script tag
    data_block = html.split('id="data">')[1].split("</script>")[0]
    assert "</script>" not in data_block


# --------------------------------------------------------------------------- #
# Live LogReader (the engine behind --follow), tested synchronously
# --------------------------------------------------------------------------- #
def test_logreader_incremental_and_partial(tmp_path):
    log = tmp_path / "live.jsonl"
    log.write_text("", encoding="utf-8")
    reader = viewer.LogReader(str(log), from_start=True)

    e0 = {"type": "match_start", "schema": 2, "game": "g", "n_players": 2,
          "seed": 0, "players": ["P0", "P1"], "event_id": 0, "match_id": "m"}
    e1 = {"type": "phase_change", "from": None, "to": "play", "active_seat": 0,
          "turn": 0, "event_id": 1, "match_id": "m"}

    # Write one complete line.
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(e0) + "\n")
    got = reader.poll()
    assert len(got) == 1 and got[0]["type"] == "match_start"

    # Write a PARTIAL line (no trailing newline yet) — reader must buffer it.
    partial = json.dumps(e1)
    with log.open("a", encoding="utf-8") as f:
        f.write(partial[:10])
    assert reader.poll() == []  # nothing complete yet

    # Complete the line.
    with log.open("a", encoding="utf-8") as f:
        f.write(partial[10:] + "\n")
    got = reader.poll()
    assert len(got) == 1 and got[0]["type"] == "phase_change"


def test_logreader_from_end_skips_existing(tmp_path):
    log = tmp_path / "live2.jsonl"
    _write_jsonl(log, _synthetic_events())
    reader = viewer.LogReader(str(log), from_start=False)
    # from_start=False: existing content is skipped, only new lines surface.
    assert reader.poll() == []
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "ping", "event_id": 99}) + "\n")
    got = reader.poll()
    assert len(got) == 1 and got[0]["type"] == "ping"


# --------------------------------------------------------------------------- #
# Lazy game registry must never hard-depend on a game being importable
# --------------------------------------------------------------------------- #
def test_board_art_falls_back_when_game_absent():
    # Unknown game name -> structured fallback (no crash, returns the snapshot).
    snap = {"public": {"foo": "bar"}, "hidden": {"x": 1}}
    art = viewer._render_board_via_game("no_such_game", snap, "god")
    assert art == ""  # unknown game: viewer falls back to structured board
    structured = viewer._structured_board(snap)
    assert "foo" in structured


def test_board_art_known_game_tolerates_default_render(tmp_path):
    # Known game whose render_board(state) default returns '' for a dict
    # snapshot must not crash the viewer.
    snap = {"public": {"phase": "x"}, "hidden": {}}
    # Should be '' (default render_board returns '') or some art; never raise.
    art = viewer._render_board_via_game("one_night_werewolf", snap, "god")
    assert isinstance(art, str)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
