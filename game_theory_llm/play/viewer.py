"""Pure JSONL-log watcher / replay tool for the play harness (spec §6).

This module is a *pure log reader*: it imports nothing from the game engine
except a lazy ``name -> Game`` registry used solely to call
``Game.render_board(snapshot, reveal=...)`` for per-game board art. If a game
class is missing (units built in parallel) or ``render_board`` raises / returns
``""``, the viewer degrades gracefully to the structured log. Everything else is
recovered from the JSONL alone, proving the log is self-sufficient (the spec §4
reconstruction guarantee).

Capabilities (driven by ``scripts/watch_match.py`` or ``python3 -m
game_theory_llm.play.viewer``):

* static god-view replay (default), paged;
* ``--follow`` tail a live match (~200 ms poll, tolerating a partial trailing
  line);
* ``--seat N`` reconstruct ONLY what seat N saw (seat N's prompt records +
  observations whose audience contains N; private messages N did not receive
  appear at most as ``message_meta``; hidden roles hidden except N's own / peeks);
* ``--god`` reveal all hidden info from ``setup.god_view`` +
  ``state_snapshot.hidden`` (the default);
* ``--step`` event stepper (n / p / q); ``--speed S`` auto-play S events/sec;
* ``--html OUT.html`` self-contained replay with a scrub bar over ``event_id``
  and a client-side seat selector;
* ``--metrics`` print the steering-metrics table via
  ``game_theory_llm.play.metrics``.

Optional dependency: ``rich`` (colour / panels). If it is unavailable the
renderer degrades to plain ANSI / UTF-8 text — nothing here hard-depends on it.
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Optional rich. Import is wrapped so an absent / broken rich never crashes the
# viewer; we only use it for nicer colour and never depend on its API shape.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised indirectly
    from rich.console import Console as _RichConsole  # type: ignore

    _HAVE_RICH = True
except Exception:  # pragma: no cover
    _RichConsole = None  # type: ignore
    _HAVE_RICH = False


# Glyphs (UTF-8); ASCII fallbacks chosen when output is not a UTF-8 tty.
_GLYPHS_UTF8 = {
    "public": "\U0001F4E2",   # loudspeaker
    "private": "\U0001F512",  # lock
    "propose": "⚑",      # flag
    "accept": "✔",       # check
    "decline": "✘",      # ballot x
    "betray": "\U0001F4A5",   # collision
    "honour": "✅",       # white check
    "arrow": "→",        # right arrow
}
_GLYPHS_ASCII = {
    "public": "[PUB]",
    "private": "[PRV]",
    "propose": "(propose)",
    "accept": "(accept)",
    "decline": "(decline)",
    "betray": "(BETRAY)",
    "honour": "(honour)",
    "arrow": "->",
}


def _supports_utf8() -> bool:
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in enc


def _glyphs(utf8: Optional[bool] = None) -> Dict[str, str]:
    if utf8 is None:
        utf8 = _supports_utf8()
    return _GLYPHS_UTF8 if utf8 else _GLYPHS_ASCII


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_events(path: str) -> List[dict]:
    """Parse a JSONL log into a list of records.

    Tolerant of a partial / un-terminated trailing line (so a live match being
    tailed with ``--follow`` never crashes the reader). Blank lines are skipped;
    any line that fails to parse is dropped (only the *last* line is normally
    partial, but we are defensive about all of them).
    """
    events: List[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Partial trailing line (or a corrupt line mid-file): skip it.
                continue
    return events


def _seat_name(players: List[str], seat: int) -> str:
    if 0 <= seat < len(players):
        return players[seat]
    if seat < 0:
        return "GOD"
    return f"P{seat}"


# --------------------------------------------------------------------------- #
# Reconstruction -> Frames
# --------------------------------------------------------------------------- #
@dataclass
class Frame:
    """One renderable point in the replay, keyed by ``event_id``.

    A frame is built for every event; it carries the *cumulative* view of the
    match up to and including that event (board snapshot, chat log so far,
    alliance ledger, god panel, per-seat masked panels). This lets the stepper
    / scrub bar jump to any event and render a consistent picture.
    """

    event_id: int
    turn: int
    phase: str
    event: dict                          # the raw record at this index
    board: str = ""                      # game art (or structured fallback)
    chats: List[dict] = field(default_factory=list)   # cumulative chat entries
    alliances: List[dict] = field(default_factory=list)  # ledger snapshot
    god_panel: str = ""                  # hidden truth
    seat_panels: Dict[int, str] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)  # honour/betray etc.
    banner: str = ""                     # phase banner text (only on changes)


class MatchReplay:
    """Folded, navigable view over a parsed JSONL match log.

    Construct via :func:`reconstruct`. ``frames`` is one entry per event; index
    them positionally (0..len-1). ``render(idx, reveal=...)`` produces a text
    panel from any vantage point: ``"god"`` (default) or ``"seatN"``.
    """

    def __init__(self, events: List[dict]) -> None:
        self.events = events
        self.start = next((e for e in events if e.get("type") == "match_start"), {})
        self.terminal = next(
            (e for e in reversed(events) if e.get("type") == "terminal"), {}
        )
        self.game: str = self.start.get("game", "?")
        self.n_players: int = int(self.start.get("n_players", 0) or 0)
        self.players: List[str] = list(self.start.get("players") or [])
        if not self.players and self.n_players:
            self.players = [f"P{i}" for i in range(self.n_players)]
        self.config: dict = self.start.get("config", {}) or {}
        self.god_view_setup: dict = {}
        self.frames: List[Frame] = []
        self._build()

    # -- helpers ------------------------------------------------------------ #
    def _board_art(self, snapshot: dict, reveal: str) -> str:
        """Try per-game board art via a lazy registry; else structured dump."""
        art = _render_board_via_game(self.game, snapshot, reveal)
        if art:
            return art
        return _structured_board(snapshot)

    def _alliance_ledger(self, alliances: Dict[int, dict]) -> List[dict]:
        out = []
        for aid in sorted(alliances):
            out.append(dict(alliances[aid]))
        return out

    # -- fold --------------------------------------------------------------- #
    def _build(self) -> None:
        chats: List[dict] = []
        alliances: Dict[int, dict] = {}     # alliance_id -> ledger entry
        counters = {"honored": 0, "betrayed": 0,
                    "proposed": 0, "accepted": 0, "broken": 0, "declined": 0}
        cur_phase = self.start.get("phase") or ""
        cur_turn = 0
        cur_snapshot: dict = {}

        for e in self.events:
            etype = e.get("type")
            banner = ""
            if etype == "setup":
                self.god_view_setup = e.get("god_view", {}) or {}
            elif etype == "state_snapshot":
                cur_snapshot = e.get("snapshot", {}) or {}
                cur_phase = e.get("phase") or cur_phase
                cur_turn = e.get("turn", cur_turn)
            elif etype == "phase_change":
                cur_phase = e.get("to") or cur_phase
                cur_turn = e.get("turn", cur_turn)
                g = _glyphs()
                banner = (f"PHASE: {e.get('from')} {g['arrow']} {e.get('to')} "
                          f"(turn {cur_turn}, active "
                          f"{_seat_name(self.players, e.get('active_seat', -1))})")
            elif etype == "observation":
                cur_turn = e.get("turn", cur_turn)
                obs = e.get("obs", {}) or {}
                audience = e.get("audience", []) or []
                ot = obs.get("type")
                if ot in ("message", "message_meta"):
                    chats.append({
                        "event_id": e.get("event_id"),
                        "turn": e.get("turn"),
                        "obs": obs,
                        "audience": list(audience),
                    })
                elif ot == "alliance_event":
                    self._apply_alliance_event(obs, alliances, counters)
            elif etype in ("prompt", "action"):
                cur_turn = e.get("turn", cur_turn)
                cur_phase = e.get("phase") or cur_phase

            frame = Frame(
                event_id=e.get("event_id", len(self.frames)),
                turn=cur_turn,
                phase=cur_phase,
                event=e,
                chats=list(chats),
                alliances=self._alliance_ledger(alliances),
                counters=dict(counters),
                banner=banner,
            )
            # Board art + god panel are derived lazily per render, but we stash
            # the snapshot so render() can reproduce them at any vantage point.
            frame.board = ""  # filled in render() using cur_snapshot
            frame._snapshot = cur_snapshot  # type: ignore[attr-defined]
            self.frames.append(frame)

    def _apply_alliance_event(self, obs: dict, alliances: Dict[int, dict],
                              counters: Dict[str, int]) -> None:
        aid = obs.get("alliance_id")
        if aid is None:
            return
        event = obs.get("event")
        entry = alliances.get(aid)
        if entry is None:
            entry = {
                "id": aid,
                "members": list(obs.get("members", [])),
                "kind": obs.get("kind", ""),
                "proposer": obs.get("proposer"),
                "status": "proposed",
                "honored": 0,
                "betrayed": 0,
                "broken_by": None,
            }
            alliances[aid] = entry
        # keep latest membership / kind
        if obs.get("members"):
            entry["members"] = list(obs["members"])
        if obs.get("kind"):
            entry["kind"] = obs["kind"]
        if event == "propose":
            entry["status"] = "proposed"
            counters["proposed"] += 1
        elif event == "accept":
            entry["status"] = "active"
            counters["accepted"] += 1
        elif event == "decline":
            entry["status"] = "declined"
            counters["declined"] += 1
        elif event == "break":
            entry["status"] = "broken"
            entry["broken_by"] = obs.get("actor")
            counters["broken"] += 1
        elif event == "honored":
            entry["honored"] += 1
            counters["honored"] += 1
        elif event == "betrayed":
            entry["status"] = "BETRAYED"
            entry["broken_by"] = obs.get("actor")
            entry["betrayed"] += 1
            counters["betrayed"] += 1
        elif event == "expired":
            entry["status"] = "expired"

    # -- masking ------------------------------------------------------------ #
    def _visible_chats(self, chats: List[dict], reveal: str) -> List[dict]:
        """Return chat entries visible from ``reveal``.

        ``reveal == "god"`` -> all entries at full content.
        ``reveal == "seatN"`` -> only entries whose audience contains N; an
        entry seat N is *not* in is dropped entirely (the metadata-leak entries
        the runner sent to bystanders, ``message_meta``, are themselves
        audience-scoped, so they survive iff N was their intended bystander).
        """
        if reveal == "god":
            return chats
        seat = _seat_from_reveal(reveal)
        if seat is None:
            return chats
        out = []
        for c in chats:
            if seat in (c.get("audience") or []):
                out.append(c)
        return out

    def _format_chat(self, c: dict, reveal: str, g: Dict[str, str]) -> str:
        obs = c["obs"]
        frm = obs.get("from")
        frm_name = _seat_name(self.players, frm) if frm is not None else "?"
        if obs.get("type") == "message_meta":
            n = obs.get("n_recipients", "?")
            return (f"  {g['private']} [{frm_name} {g['arrow']} ?] "
                    f"(private to {n}, content hidden)")
        scope = obs.get("scope")
        text = obs.get("text", "")
        if scope == "public":
            return f"  {g['public']} [PUBLIC] {frm_name}: {text}"
        to = obs.get("to", []) or []
        to_names = ", ".join(_seat_name(self.players, t) for t in to)
        god_tag = " (god-only)" if reveal == "god" else ""
        return (f"  {g['private']} [{frm_name} {g['arrow']} {to_names} whisper]"
                f"{god_tag}: {text}")

    def _format_alliance_ledger(self, ledger: List[dict],
                                g: Dict[str, str]) -> List[str]:
        lines = []
        for a in ledger:
            members = "{" + ",".join(
                _seat_name(self.players, m) for m in a.get("members", [])) + "}"
            status = a.get("status", "?")
            tag = ""
            if status == "BETRAYED":
                by = _seat_name(self.players, a.get("broken_by", -1))
                tag = f" {g['betray']} BETRAYED by {by}"
            elif status == "active":
                tag = " active"
            elif status == "broken":
                by = _seat_name(self.players, a.get("broken_by", -1))
                tag = f" broken by {by}"
            elif status == "declined":
                tag = " declined"
            elif status == "proposed":
                tag = " proposed"
            extra = []
            if a.get("honored"):
                extra.append(f"{g['honour']}{a['honored']}")
            if a.get("betrayed") and status != "BETRAYED":
                extra.append(f"{g['betray']}{a['betrayed']}")
            extra_s = (" " + " ".join(extra)) if extra else ""
            lines.append(f"  #{a['id']} {members} {a.get('kind','')}"
                         f"{tag}{extra_s}")
        return lines

    def _god_panel(self, frame: Frame) -> str:
        snap = getattr(frame, "_snapshot", {}) or {}
        hidden = snap.get("hidden", {}) if isinstance(snap, dict) else {}
        parts = []
        if self.god_view_setup:
            parts.append("setup.god_view: " + json.dumps(
                self.god_view_setup, default=str, sort_keys=True))
        if hidden:
            parts.append("snapshot.hidden: " + json.dumps(
                hidden, default=str, sort_keys=True))
        return "\n".join(parts)

    # -- public render ------------------------------------------------------ #
    def render(self, idx: int, *, reveal: str = "god") -> str:
        """Render frame ``idx`` from vantage point ``reveal`` (``"god"`` or
        ``"seatN"``) as a plain-text panel (UTF-8 glyphs where supported)."""
        if not self.frames:
            return "(empty log)"
        idx = max(0, min(idx, len(self.frames) - 1))
        frame = self.frames[idx]
        g = _glyphs()
        snap = getattr(frame, "_snapshot", {}) or {}
        seat = _seat_from_reveal(reveal)

        lines: List[str] = []
        title = f"=== {self.game} | event {frame.event_id} | turn {frame.turn} | phase {frame.phase} ==="
        if reveal != "god" and seat is not None:
            title += f"  (view: {_seat_name(self.players, seat)})"
        else:
            title += "  (view: GOD)"
        lines.append(title)

        if frame.banner:
            lines.append(">>> " + frame.banner)

        # The event itself (so the stepper shows *what* just happened).
        lines.append(self._describe_event(frame.event, reveal, g))

        # Board art.
        board = self._board_art(snap, reveal if reveal != "god" else "god")
        if board:
            lines.append("-- board --")
            lines.append(board)

        # Chat log (masked).
        vchats = self._visible_chats(frame.chats, reveal)
        if vchats:
            lines.append("-- chat --")
            for c in vchats[-12:]:
                lines.append(self._format_chat(c, reveal, g))

        # Alliance ledger.
        if frame.alliances:
            lines.append("-- alliances --")
            lines.extend(self._format_alliance_ledger(frame.alliances, g))

        # God panel (only in god view).
        if reveal == "god":
            gp = self._god_panel(frame)
            if gp:
                lines.append("-- god panel --")
                lines.append(gp)

        # Footer counters.
        c = frame.counters
        lines.append(
            f"-- footer -- {g['honour']} honoured={c.get('honored',0)}  "
            f"{g['betray']} betrayed={c.get('betrayed',0)}  "
            f"proposed={c.get('proposed',0)}  accepted={c.get('accepted',0)}  "
            f"broken={c.get('broken',0)}")
        return "\n".join(lines)

    def _describe_event(self, e: dict, reveal: str, g: Dict[str, str]) -> str:
        etype = e.get("type")
        seat = _seat_from_reveal(reveal)
        if etype == "prompt":
            # In a seat view, only show the seat's own prompts verbatim.
            if reveal != "god" and seat is not None and e.get("player") != seat:
                return f"[event {e.get('event_id')}] prompt -> P{e.get('player')} (not shown)"
            return (f"[event {e.get('event_id')}] PROMPT "
                    f"{_seat_name(self.players, e.get('player', -1))}:\n"
                    f"{_indent(e.get('prompt',''))}")
        if etype == "action":
            return (f"[event {e.get('event_id')}] ACTION "
                    f"{_seat_name(self.players, e.get('player', -1))}: "
                    f"{json.dumps(e.get('action', {}), default=str)}")
        if etype == "observation":
            obs = e.get("obs", {}) or {}
            ot = obs.get("type")
            aud = e.get("audience", [])
            # In a seat view, hide observations the seat is not entitled to.
            if reveal != "god" and seat is not None and seat not in (aud or []):
                return f"[event {e.get('event_id')}] observation (not visible to you)"
            return (f"[event {e.get('event_id')}] OBS {ot} "
                    f"audience={aud}")
        if etype == "phase_change":
            return f"[event {e.get('event_id')}] phase_change -> {e.get('to')}"
        if etype == "state_snapshot":
            return f"[event {e.get('event_id')}] state_snapshot phase={e.get('phase')}"
        if etype == "terminal":
            return (f"[event {e.get('event_id')}] TERMINAL winner="
                    f"{e.get('winner')} reason={e.get('win_reason','')!r} "
                    f"rewards={e.get('rewards')}")
        if etype == "match_start":
            return (f"[event {e.get('event_id')}] MATCH START {self.game} "
                    f"n_players={self.n_players} seed={e.get('seed')}")
        if etype == "setup":
            return f"[event {e.get('event_id')}] setup (god_view recorded)"
        return f"[event {e.get('event_id')}] {etype}"


def reconstruct(events: List[dict]) -> MatchReplay:
    """Fold a list of parsed records into an ordered, navigable replay."""
    return MatchReplay(events)


def _seat_from_reveal(reveal: str) -> Optional[int]:
    if not reveal or reveal == "god":
        return None
    if reveal.startswith("seat"):
        try:
            return int(reveal[4:])
        except ValueError:
            return None
    try:
        return int(reveal)
    except (TypeError, ValueError):
        return None


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + ln for ln in str(text).splitlines())


def _structured_board(snapshot: dict) -> str:
    """Fallback board: pretty-print the public part of the snapshot."""
    if not snapshot:
        return ""
    public = snapshot.get("public", snapshot) if isinstance(snapshot, dict) else snapshot
    try:
        return json.dumps(public, default=str, indent=2, sort_keys=True)
    except Exception:
        return str(public)


# --------------------------------------------------------------------------- #
# Lazy per-game board art (the one game-specific concession; never hard-deps)
# --------------------------------------------------------------------------- #
# Maps the logged ``game`` name to (module path, class name). The viewer
# imports lazily and tolerates ImportError (units built in parallel).
_GAME_REGISTRY = {
    "one_night_werewolf": ("game_theory_llm.play.games.one_night_werewolf",
                           "OneNightWerewolf"),
    "secret_hitler": ("game_theory_llm.play.games.secret_hitler", "SecretHitler"),
    "risk_lite": ("game_theory_llm.play.games.risk_lite", "RiskLite"),
    "risk": ("game_theory_llm.play.games.risk_lite", "RiskLite"),
    "diplomacy_lite": ("game_theory_llm.play.games.diplomacy_lite",
                       "DiplomacyLite"),
    "diplomacy": ("game_theory_llm.play.games.diplomacy_lite", "DiplomacyLite"),
    "monopoly_lite": ("game_theory_llm.play.games.monopoly_lite", "MonopolyLite"),
    "monopoly": ("game_theory_llm.play.games.monopoly_lite", "MonopolyLite"),
}

_GAME_CLASS_CACHE: Dict[str, Any] = {}


def _lookup_game_class(game: str):
    if game in _GAME_CLASS_CACHE:
        return _GAME_CLASS_CACHE[game]
    spec = _GAME_REGISTRY.get(game)
    cls = None
    if spec is not None:
        mod_path, cls_name = spec
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
        except Exception:
            cls = None
    _GAME_CLASS_CACHE[game] = cls
    return cls


def _render_board_via_game(game: str, snapshot: dict, reveal: str) -> str:
    """Call ``Game.render_board(snapshot, reveal=...)`` if the game class is
    importable and provides art; otherwise return ``""``.

    The game's ``render_board`` expects a state object; the viewer only has the
    logged snapshot dict, so we pass that. Games that cannot render from a bare
    snapshot simply return ``""`` (or raise, which we swallow), and the caller
    falls back to the structured board. This keeps the viewer game-agnostic.
    """
    if not snapshot:
        return ""
    cls = _lookup_game_class(game)
    if cls is None:
        return ""
    try:
        inst = cls()
    except Exception:
        return ""
    fn = getattr(inst, "render_board", None)
    if fn is None:
        return ""
    try:
        art = fn(snapshot, reveal=reveal)
    except Exception:
        return ""
    return art or ""


# --------------------------------------------------------------------------- #
# Live tailing
# --------------------------------------------------------------------------- #
class LogReader:
    """Incremental JSONL reader for ``--follow``.

    Each :meth:`poll` returns any newly-completed records since the last call,
    buffering a partial trailing line until its newline arrives. Works against a
    live match because the runner flushes per ``log()`` (open / append / close).
    """

    def __init__(self, path: str, *, from_start: bool = True) -> None:
        self.path = Path(path)
        self._buf = ""
        self._pos = 0
        if not from_start and self.path.exists():
            self._pos = self.path.stat().st_size

    def poll(self) -> List[dict]:
        records: List[dict] = []
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(self._pos)
            chunk = f.read()
            self._pos = f.tell()
        if not chunk:
            return records
        self._buf += chunk
        # Split out only complete lines; keep the trailing partial in the buffer.
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Should not happen for a complete line, but stay defensive.
                continue
        return records


# --------------------------------------------------------------------------- #
# Terminal renderer
# --------------------------------------------------------------------------- #
class TerminalRenderer:
    def __init__(self, *, use_rich: Optional[bool] = None) -> None:
        if use_rich is None:
            use_rich = _HAVE_RICH and _supports_utf8()
        self.use_rich = use_rich and _HAVE_RICH
        self._console = _RichConsole() if (self.use_rich and _RichConsole) else None

    def emit(self, text: str) -> None:
        if self._console is not None:  # pragma: no cover - rich path
            try:
                self._console.print(text, highlight=False, markup=False)
                return
            except Exception:
                pass
        try:
            print(text)
        except UnicodeEncodeError:  # pragma: no cover
            print(text.encode("ascii", "replace").decode("ascii"))

    # -- modes -------------------------------------------------------------- #
    def static(self, replay: MatchReplay, *, reveal: str = "god") -> None:
        for idx in range(len(replay.frames)):
            f = replay.frames[idx]
            # Only print frames that advance the narrative (avoid one panel per
            # prompt+obs spam): show match_start, phase changes, actions,
            # observations, terminal. We render every event but skip pure
            # snapshot duplicates from cluttering the static view.
            etype = f.event.get("type")
            if etype in ("state_snapshot",):
                continue
            self.emit(replay.render(idx, reveal=reveal))
            self.emit("")

    def step(self, replay: MatchReplay, *, reveal: str = "god",
             inp=input) -> None:  # pragma: no cover - interactive
        idx = 0
        n = len(replay.frames)
        while True:
            self.emit(replay.render(idx, reveal=reveal))
            self.emit(f"[{idx + 1}/{n}] (n)ext / (p)rev / (q)uit > ")
            try:
                cmd = inp().strip().lower()
            except EOFError:
                break
            if cmd in ("q", "quit"):
                break
            if cmd in ("p", "prev"):
                idx = max(0, idx - 1)
            else:
                idx = min(n - 1, idx + 1)
                if idx == n - 1 and cmd in ("n", "next", ""):
                    # allow stepping past the end once, then stop
                    pass

    def autoplay(self, replay: MatchReplay, *, reveal: str = "god",
                 speed: float = 2.0) -> None:  # pragma: no cover - timing
        delay = 1.0 / speed if speed > 0 else 0.0
        for idx in range(len(replay.frames)):
            etype = replay.frames[idx].event.get("type")
            if etype == "state_snapshot":
                continue
            self.emit(replay.render(idx, reveal=reveal))
            self.emit("")
            if delay:
                time.sleep(delay)

    def follow(self, path: str, *, reveal: str = "god", from_start: bool = True,
               poll_s: float = 0.2,
               max_idle_polls: Optional[int] = None) -> None:  # pragma: no cover
        reader = LogReader(path, from_start=from_start)
        seen: List[dict] = []
        idle = 0
        while True:
            new = reader.poll()
            if new:
                idle = 0
                seen.extend(new)
                replay = reconstruct(seen)
                # render only the freshly arrived frames
                start = len(replay.frames) - len(new)
                for idx in range(max(0, start), len(replay.frames)):
                    if replay.frames[idx].event.get("type") == "state_snapshot":
                        continue
                    self.emit(replay.render(idx, reveal=reveal))
                    self.emit("")
                if any(e.get("type") == "terminal" for e in new):
                    break
            else:
                idle += 1
                if max_idle_polls is not None and idle >= max_idle_polls:
                    break
            time.sleep(poll_s)


# --------------------------------------------------------------------------- #
# HTML renderer (self-contained, client-side seat selector + scrub bar)
# --------------------------------------------------------------------------- #
class HtmlRenderer:
    def render(self, replay: MatchReplay) -> str:
        """Build a single self-contained HTML file: the JSONL embedded as
        ``<script type="application/json">``, a scrub bar over ``event_id``, and
        a client-side seat selector that re-masks chat/observations in the
        browser. A final scoreboard + alliance summary closes it."""
        payload = {
            "game": replay.game,
            "n_players": replay.n_players,
            "players": replay.players,
            "config": replay.config,
            "events": replay.events,
            "terminal": replay.terminal,
            "god_view_setup": replay.god_view_setup,
        }
        data_json = json.dumps(payload, default=str)
        # Escape only the closing-script sequence so the embedded JSON is safe.
        data_json = data_json.replace("</", "<\\/")
        seat_opts = "".join(
            f'<option value="{i}">{_html.escape(replay.players[i] if i < len(replay.players) else f"P{i}")}</option>'
            for i in range(replay.n_players))
        title = _html.escape(f"Replay: {replay.game}")
        return _HTML_TEMPLATE.format(
            title=title, seat_opts=seat_opts, data_json=data_json,
            n_events=len(replay.events))


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body{{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1117;color:#c9d1d9;margin:0;padding:1rem;}}
 h1{{font-size:1.1rem;color:#58a6ff;}}
 .bar{{position:sticky;top:0;background:#161b22;padding:.5rem;border:1px solid #30363d;border-radius:6px;}}
 .panel{{white-space:pre-wrap;border:1px solid #30363d;border-radius:6px;padding:.75rem;margin-top:.5rem;background:#161b22;}}
 .pub{{color:#7ee787;}} .prv{{color:#d2a8ff;}} .betray{{color:#ff7b72;font-weight:bold;}}
 .honour{{color:#7ee787;}} .banner{{color:#f2cc60;font-weight:bold;}}
 .ally{{color:#79c0ff;}} .god{{color:#8b949e;}}
 select,input{{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;}}
 label{{margin-right:1rem;}}
</style></head><body>
<h1>{title}</h1>
<div class="bar">
 <label>Event <span id="eid">0</span>/{n_events}
  <input id="scrub" type="range" min="0" max="{n_events}" value="{n_events}" style="width:50%"></label>
 <label>View
  <select id="seat"><option value="god" selected>GOD (all)</option>{seat_opts}</select></label>
</div>
<div class="panel" id="out"></div>
<div class="panel god" id="score"></div>
<script type="application/json" id="data">{data_json}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const players = DATA.players || [];
function nm(s){{ if(s<0) return 'GOD'; return players[s]||('P'+s); }}
function seatOf(reveal){{ return reveal==='god'?null:parseInt(reveal,10); }}
// Fold events up to index `upto`, masked by `reveal`.
function fold(upto, reveal){{
  const seat = seatOf(reveal);
  let phase='', turn=0, chats=[], alliances={{}}, counters={{honored:0,betrayed:0,proposed:0,accepted:0,broken:0,declined:0}};
  let lines=[], setupGod=null, hidden=null;
  for(let i=0;i<=upto && i<DATA.events.length;i++){{
    const e=DATA.events[i], t=e.type;
    if(t==='setup'){{ setupGod=e.god_view; }}
    else if(t==='state_snapshot'){{ phase=e.phase||phase; turn=e.turn; hidden=(e.snapshot||{{}}).hidden||null; }}
    else if(t==='phase_change'){{ phase=e.to||phase; turn=e.turn; if(i===upto) lines.push('<span class="banner">&gt;&gt;&gt; PHASE: '+e.from+' -> '+e.to+'</span>'); }}
    else if(t==='observation'){{
      const obs=e.obs||{{}}, aud=e.audience||[];
      if(obs.type==='message'||obs.type==='message_meta') chats.push({{obs:obs,aud:aud}});
      else if(obs.type==='alliance_event') applyAlly(obs,alliances,counters);
    }}
    else if(t==='prompt'||t==='action'){{ phase=e.phase||phase; turn=e.turn; }}
  }}
  // current event description
  const cur=DATA.events[Math.min(upto,DATA.events.length-1)]||{{}};
  lines.push('<b>event '+ (cur.event_id) +' | turn '+turn+' | phase '+phase+' | view '+(seat===null?'GOD':nm(seat))+'</b>');
  lines.push(descEvent(cur, seat));
  // chat (masked)
  const vis = chats.filter(c=> seat===null || (c.aud||[]).includes(seat));
  if(vis.length){{ lines.push('-- chat --'); vis.slice(-15).forEach(c=>lines.push(fmtChat(c,reveal))); }}
  // alliances
  const ledger=Object.keys(alliances).sort((a,b)=>a-b).map(k=>alliances[k]);
  if(ledger.length){{ lines.push('-- alliances --'); ledger.forEach(a=>lines.push(fmtAlly(a))); }}
  if(reveal==='god'){{
    let gp=[]; if(setupGod) gp.push('setup.god_view: '+JSON.stringify(setupGod));
    if(hidden) gp.push('snapshot.hidden: '+JSON.stringify(hidden));
    if(gp.length){{ lines.push('-- god panel --'); lines.push('<span class="god">'+gp.join('\\n')+'</span>'); }}
  }}
  lines.push('-- footer -- <span class="honour">honoured='+counters.honored+'</span> <span class="betray">betrayed='+counters.betrayed+'</span> proposed='+counters.proposed+' accepted='+counters.accepted+' broken='+counters.broken);
  return lines.join('\\n');
}}
function descEvent(e, seat){{
  if(!e||!e.type) return '';
  if(e.type==='prompt'){{ if(seat!==null && e.player!==seat) return '[prompt -> '+nm(e.player)+' (hidden)]'; return 'PROMPT '+nm(e.player)+':\\n'+(e.prompt||''); }}
  if(e.type==='action') return 'ACTION '+nm(e.player)+': '+JSON.stringify(e.action||{{}});
  if(e.type==='observation'){{ const aud=e.audience||[]; if(seat!==null && !aud.includes(seat)) return '[observation not visible to you]'; return 'OBS '+((e.obs||{{}}).type)+' audience='+JSON.stringify(aud); }}
  if(e.type==='terminal') return 'TERMINAL winner='+e.winner+' reason='+JSON.stringify(e.win_reason||'')+' rewards='+JSON.stringify(e.rewards||[]);
  if(e.type==='match_start') return 'MATCH START '+DATA.game+' seed='+e.seed;
  return '['+e.type+']';
}}
function fmtChat(c,reveal){{
  const o=c.obs, frm=nm(o.from);
  if(o.type==='message_meta') return '  <span class="prv">[private from '+frm+', content hidden]</span>';
  if(o.scope==='public') return '  <span class="pub">[PUBLIC] '+frm+': '+esc(o.text||'')+'</span>';
  const to=(o.to||[]).map(nm).join(', ');
  return '  <span class="prv">['+frm+' -> '+to+' whisper'+(reveal==='god'?' (god-only)':'')+']: '+esc(o.text||'')+'</span>';
}}
function fmtAlly(a){{
  const mem='{{'+(a.members||[]).map(nm).join(',')+'}}';
  let tag=a.status; if(a.status==='BETRAYED') tag='<span class="betray">BETRAYED by '+nm(a.broken_by)+'</span>';
  let extra=''; if(a.honored) extra+=' <span class="honour">honoured x'+a.honored+'</span>';
  return '  <span class="ally">#'+a.id+' '+mem+' '+(a.kind||'')+' '+tag+extra+'</span>';
}}
function applyAlly(obs,alliances,counters){{
  const aid=obs.alliance_id; if(aid===undefined||aid===null) return;
  let e=alliances[aid]; if(!e){{ e={{id:aid,members:obs.members||[],kind:obs.kind||'',status:'proposed',honored:0,betrayed:0,broken_by:null}}; alliances[aid]=e; }}
  if(obs.members) e.members=obs.members; if(obs.kind) e.kind=obs.kind;
  const ev=obs.event;
  if(ev==='propose'){{e.status='proposed';counters.proposed++;}}
  else if(ev==='accept'){{e.status='active';counters.accepted++;}}
  else if(ev==='decline'){{e.status='declined';counters.declined++;}}
  else if(ev==='break'){{e.status='broken';e.broken_by=obs.actor;counters.broken++;}}
  else if(ev==='honored'){{e.honored++;counters.honored++;}}
  else if(ev==='betrayed'){{e.status='BETRAYED';e.broken_by=obs.actor;e.betrayed++;counters.betrayed++;}}
}}
function esc(s){{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}
function scoreboard(){{
  const t=DATA.terminal||{{}}; const s=t.alliance_summary||{{}};
  let o='SCOREBOARD  winner='+t.winner+'  rewards='+JSON.stringify(t.rewards||[])+'\\n';
  o+='alliances: proposed='+(s.n_proposed||0)+' accepted='+(s.n_accepted||0)+' honoured='+(s.n_honored||0)+' betrayed='+(s.n_betrayed||0);
  return o;
}}
const out=document.getElementById('out'), score=document.getElementById('score');
const scrub=document.getElementById('scrub'), seatSel=document.getElementById('seat'), eid=document.getElementById('eid');
function refresh(){{ const i=Math.min(parseInt(scrub.value,10),DATA.events.length-1); eid.textContent=i; out.innerHTML=fold(i,seatSel.value); score.textContent=scoreboard(); }}
scrub.addEventListener('input',refresh); seatSel.addEventListener('change',refresh);
refresh();
</script></body></html>"""


# --------------------------------------------------------------------------- #
# Metrics passthrough
# --------------------------------------------------------------------------- #
def metrics_table(path: str) -> str:
    """Render the steering-metrics table for a single log via
    ``game_theory_llm.play.metrics`` (a pure log reduction; spec §4).

    Returns a plain-text table including the alliance summary. Falls back to a
    hand-rolled table if pandas is unavailable so ``--metrics`` always works.
    """
    from game_theory_llm.play import metrics as _metrics

    m = _metrics.load_match(path)
    lines: List[str] = []
    lines.append("=== Steering metrics ===")
    lines.append(f"game={m.get('game')}  match_id={m.get('match_id')}  "
                 f"seed={m.get('seed')}  winner={m.get('winner')}")
    tags = m.get("steering_tags") or []
    lines.append(f"steering_tags={tags}")
    lines.append("")
    lines.append("-- alliance summary --")
    alli = m.get("alliances", {}) or {}
    for k in ("n_proposed", "n_accepted", "n_declined", "n_broken",
              "n_honored", "n_betrayed"):
        lines.append(f"  {k:<12} {alli.get(k, 0)}")
    per = alli.get("per_player", {}) or {}
    if per:
        lines.append("  per_player:")
        hdr = "    seat  proposed accepted honored betrayed betrayed_against"
        lines.append(hdr)
        for seat in sorted(per, key=lambda s: int(s)):
            d = per[seat]
            lines.append(
                f"    {seat:<5} {d.get('proposed',0):<8} {d.get('accepted',0):<8} "
                f"{d.get('honored',0):<7} {d.get('betrayed',0):<8} "
                f"{d.get('betrayed_against',0)}")
    lines.append("")
    lines.append("-- rates --")
    rates = m.get("rates", {}) or {}
    for k in ("formation", "betrayal", "honour"):
        v = rates.get(k)
        lines.append(f"  {k:<10} {'n/a' if v is None else f'{v:.3f}'}")
    msgs = m.get("messages", {}) or {}
    lines.append("")
    lines.append("-- messages --")
    pr = msgs.get("private_ratio")
    lines.append(f"  total={msgs.get('total',0)}  public={msgs.get('public',0)}  "
                 f"private={msgs.get('private',0)}  meta={msgs.get('meta',0)}  "
                 f"private_ratio={'n/a' if pr is None else f'{pr:.3f}'}")
    fbt = m.get("first_betrayal_turn")
    lines.append(f"  first_betrayal_turn={fbt}")

    # Try to also append the multi-match DataFrame table when pandas exists.
    try:
        df = _metrics.steering_table([path])
        lines.append("")
        lines.append("-- steering_table (1 row) --")
        lines.append(df.to_string(index=False))
    except Exception:
        pass
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watch_match",
        description="Watch / replay a play-harness JSONL match log (spec §6).")
    p.add_argument("log", help="path to the JSONL match log")
    p.add_argument("--follow", action="store_true",
                   help="tail a live match (~200ms poll)")
    p.add_argument("--from-start", action="store_true",
                   help="with --follow, replay from the start before tailing")
    p.add_argument("--seat", type=int, default=None,
                   help="reconstruct ONLY what seat N saw (masked view)")
    p.add_argument("--god", action="store_true",
                   help="reveal all hidden info (default vantage)")
    p.add_argument("--step", action="store_true",
                   help="interactive event stepper (n/p/q)")
    p.add_argument("--speed", type=float, default=None,
                   help="auto-play S events/sec")
    p.add_argument("--html", metavar="OUT.html", default=None,
                   help="write a self-contained HTML replay and exit")
    p.add_argument("--metrics", action="store_true",
                   help="print the steering-metrics table and exit")
    p.add_argument("--no-rich", action="store_true",
                   help="force plain ANSI output (disable rich)")
    return p


def _reveal_for(args) -> str:
    if args.seat is not None:
        return f"seat{args.seat}"
    return "god"


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    reveal = _reveal_for(args)

    if args.metrics:
        print(metrics_table(args.log))
        return 0

    if args.html:
        events = load_events(args.log)
        replay = reconstruct(events)
        Path(args.html).write_text(HtmlRenderer().render(replay),
                                   encoding="utf-8")
        print(f"Wrote HTML replay: {args.html}")
        return 0

    renderer = TerminalRenderer(use_rich=False if args.no_rich else None)

    if args.follow:
        renderer.follow(args.log, reveal=reveal, from_start=args.from_start)
        return 0

    events = load_events(args.log)
    replay = reconstruct(events)

    if args.step:
        renderer.step(replay, reveal=reveal)
    elif args.speed:
        renderer.autoplay(replay, reveal=reveal, speed=args.speed)
    else:
        renderer.static(replay, reveal=reveal)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
