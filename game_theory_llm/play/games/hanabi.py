"""Hanabi (2-5 players, rules-faithful) — the pure-cooperation control game.

Hanabi is the complement to the mixed-motive games in this harness. Every seat
shares one payoff, defection is not representable, and the only channel between
players is a costly, rationed clue. That makes it the clean control for the
steering question: a cooperation vector that changes behaviour *here* is moving
coordination **competence**, not cooperative **intent**, because there is no
intent to move — nobody in Hanabi has anything to gain by defecting.

Rules implemented (standard/official, matching the Hanabi Learning Environment
conventions used by the Hanabi Challenge benchmark, Bard et al. 2020):

- **5 colours** (red/yellow/green/blue/white) x ranks with counts 1,1,1,2,2,3,3,4,4,5
  = 50 cards. Hand size 5 at 2-3 players, 4 at 4-5 players.
- **8 clue tokens, 3 fuse tokens.** A clue costs one token; a discard returns
  one (capped at 8); completing a firework (playing a 5) returns one.
- **Clues** name one colour or one rank, must be given to another seat, and
  must touch at least one card in that seat's hand (no empty clues). A clue
  marks every matching card AND implicitly marks the non-matching cards as
  *not* that colour/rank — both directions are recorded in the receiver's
  knowledge and are visible in their prompt.
- **Play**: correct next rank advances that firework; otherwise the card is
  discarded and a fuse is spent. **Discard** is illegal at 8 clue tokens.
- **End**: three fuses spent, all five fireworks complete (25), or one final
  round after the deck empties (each seat takes exactly one more turn).
- **Scoring**: ``strict_bombs=True`` (default, the benchmark convention) scores
  a bombed game 0; the un-bombed firework total is always recorded separately
  as ``fireworks_score`` so both conventions can be analysed offline.
- **``n_colors``** (default 5, the full game) shrinks the suit set. This is a
  measurement knob, not a rule change: the full game may pin a small model at
  a score of ~0, and a floor is indistinguishable from a true null, so the
  variant exists to find a regime with real dynamic range. Max score is
  ``5 * n_colors`` and rewards are normalised against it.

Hidden information is exactly one thing: a seat never sees its own hand.
``render_prompt`` masks it and substitutes that seat's clue knowledge.

Action types emitted by ``parse_action``:
  - {"type": "play",    "index": int}                 # 0-based hand slot
  - {"type": "discard", "index": int}                 # 0-based hand slot
  - {"type": "clue",    "to": int, "kind": "color"|"rank", "value": str|int}

There is no negotiation phase: in Hanabi, table talk is the thing the game
forbids, so ``MessagingMixin`` is deliberately NOT composed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..base import Action, Game, Obs, ParseError
from ..config import GameConfig

COLORS = ["red", "yellow", "green", "blue", "white"]
COLOR_ABBR = {c[0]: c for c in COLORS}          # r/y/g/b/w
RANK_COUNTS = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}
MAX_CLUES = 8
N_FUSES = 3
MAX_RANK = 5

PH_PLAY = "play"
PH_TERMINAL = "done"

# Hand size by player count (official).
HAND_SIZE_BY_N = {2: 5, 3: 5, 4: 4, 5: 4}


def _card(color: str, rank: int) -> dict:
    return {"color": color, "rank": int(rank)}


def _fmt(card: Optional[dict]) -> str:
    if not card:
        return "--"
    return f"{card['color']}{card['rank']}"


def _fresh_knowledge() -> dict:
    """What a seat knows about one of its own cards."""
    return {"color": None, "rank": None, "not_colors": [], "not_ranks": []}


@dataclass
class HanabiState:
    n_players: int
    hands: List[List[dict]]
    knowledge: List[List[dict]]          # parallel to hands; owner's view
    deck: List[dict]
    discards: List[dict] = field(default_factory=list)
    fireworks: Dict[str, int] = field(default_factory=dict)
    clue_tokens: int = MAX_CLUES
    fuse_tokens: int = N_FUSES
    to_move: int = 0
    phase: str = PH_PLAY
    turn: int = 0
    final_turns_left: Optional[int] = None   # set when the deck empties
    clue_log: List[dict] = field(default_factory=list)
    last_event: Optional[dict] = None
    winner: Optional[str] = None
    win_reason: str = ""
    end_reason: str = ""

    # ------------------------------------------------------------------ score
    @property
    def fireworks_score(self) -> int:
        return sum(self.fireworks.values())


class Hanabi(Game):
    """Cooperative-ceiling control game. See module docstring."""

    name = "hanabi"

    def __init__(self, config: "Optional[GameConfig]" = None, n_players: int = 5,
                 *, strict_bombs: bool = True, n_colors: int = 5) -> None:
        super().__init__(config)
        if n_players not in HAND_SIZE_BY_N:
            raise ValueError(f"Hanabi supports 2-5 players, got {n_players}")
        if not 2 <= n_colors <= len(COLORS):
            raise ValueError(f"n_colors must be 2..{len(COLORS)}, got {n_colors}")
        self.n_players = n_players
        self.hand_size = HAND_SIZE_BY_N[n_players]
        self.strict_bombs = bool(strict_bombs)
        # Suit count is tunable because the full 5-suit game may sit on the
        # floor for a small model — a score pinned near 0 measures nothing, and
        # a competence-floor null is indistinguishable from a true null. Fewer
        # suits is a standard variant and lifts the baseline into a range where
        # an effect is detectable at all.
        self.colors = list(COLORS[:n_colors])
        self.max_score = MAX_RANK * len(self.colors)

    # --------------------------------------------------------------- setup
    def initial_state(self, rng) -> HanabiState:
        deck = [_card(c, r) for c in self.colors for r, n in RANK_COUNTS.items()
                for _ in range(n)]
        rng.shuffle(deck)
        hands: List[List[dict]] = []
        knowledge: List[List[dict]] = []
        for _ in range(self.n_players):
            hands.append([deck.pop() for _ in range(self.hand_size)])
            knowledge.append([_fresh_knowledge() for _ in range(self.hand_size)])
        return HanabiState(
            n_players=self.n_players,
            hands=hands,
            knowledge=knowledge,
            deck=deck,
            fireworks={c: 0 for c in self.colors},
        )

    # ---------------------------------------------------------------- turns
    def active_player(self, state: HanabiState) -> int:
        if self.is_terminal(state):
            return -1
        return state.to_move

    # -------------------------------------------------------- legal actions
    def legal_actions(self, state: HanabiState, player: int) -> List[Action]:
        if self.is_terminal(state) or player != state.to_move:
            return []
        acts: List[Action] = []
        for i in range(len(state.hands[player])):
            acts.append({"type": "play", "index": i})
        if state.clue_tokens < MAX_CLUES:
            for i in range(len(state.hands[player])):
                acts.append({"type": "discard", "index": i})
        if state.clue_tokens > 0:
            for other in range(self.n_players):
                if other == player:
                    continue
                hand = state.hands[other]
                for c in sorted({card["color"] for card in hand}):
                    acts.append({"type": "clue", "to": other,
                                 "kind": "color", "value": c})
                for r in sorted({card["rank"] for card in hand}):
                    acts.append({"type": "clue", "to": other,
                                 "kind": "rank", "value": r})
        return acts

    # ------------------------------------------------------------- prompting
    def render_prompt(self, state: HanabiState, player: int) -> str:
        lines = [
            "=== HANABI ===",
            f"You are player P{player} of {self.n_players}. Hanabi is fully "
            "cooperative: every player wins or loses together on ONE shared "
            "score. There are no opposing sides.",
            "",
            self._board_text(state),
            "",
            "--- other players' hands (you can see these; they cannot) ---",
        ]
        for other in range(self.n_players):
            if other == player:
                continue
            cards = " ".join(
                f"[{i + 1}]{_fmt(c)}" for i, c in enumerate(state.hands[other]))
            hints = self._knowledge_line(state, other)
            lines.append(f"P{other}: {cards}")
            lines.append(f"     they know: {hints}")
        lines += [
            "",
            "--- YOUR hand (hidden from you; only what clues told you) ---",
            f"P{player}: {self._knowledge_line(state, player)}",
        ]
        if state.clue_log:
            lines += ["", "--- recent clues ---"]
            for e in state.clue_log[-8:]:
                lines.append(
                    f"  turn {e['turn']}: P{e['from']} told P{e['to']} "
                    f"\"{e['value']}\" -> slots "
                    f"{[i + 1 for i in e['touched']] or 'none'}")
        lines += [
            "",
            "--- your options ---",
            f"  <play>N</play>     play card in slot N (1..{len(state.hands[player])})",
        ]
        if state.clue_tokens < MAX_CLUES:
            lines.append(
                f"  <discard>N</discard>  discard slot N, regain 1 clue token")
        else:
            lines.append("  (discard is ILLEGAL right now: clue tokens are full at 8)")
        if state.clue_tokens > 0:
            lines.append("  <clue>Pk red</clue> or <clue>Pk 3</clue>   "
                         "spend 1 clue token to tell another player about a "
                         "colour or a rank")
            lines.append("  IMPORTANT: a clue gives information TO the player "
                         "you name. It tells you nothing about your own hand. "
                         "You learn about your hand only when someone clues YOU.")
            lines.append("  These are the only legal clues right now (any other "
                         "would touch no card and be rejected):")
            lines.append(self._legal_clue_text(state, player))
        else:
            lines.append("  (clues are ILLEGAL right now: 0 clue tokens left — "
                         "discard to regain one)")
        lines += [
            "",
            "Reply with EXACTLY ONE tag and nothing else.",
        ]
        return "\n".join(lines)

    def _board_text(self, state: HanabiState) -> str:
        fw = "  ".join(f"{c}:{state.fireworks[c]}" for c in self.colors)
        disc = " ".join(_fmt(c) for c in state.discards[-12:]) or "(none)"
        parts = [
            f"fireworks: {fw}   (score {state.fireworks_score}/{self.max_score})",
            f"clue tokens: {state.clue_tokens}/{MAX_CLUES}    "
            f"fuses left: {state.fuse_tokens}/{N_FUSES}    "
            f"deck: {len(state.deck)} cards",
            f"discard pile (recent): {disc}",
        ]
        if state.final_turns_left is not None:
            parts.append(f"DECK EMPTY — final round: {state.final_turns_left} "
                         "turn(s) remain in the whole game.")
        return "\n".join(parts)

    def _knowledge_line(self, state: HanabiState, seat: int) -> str:
        """One slot per line, spelled out. The compact form (``[1]red?(not 3)``)
        was ambiguous in both directions — ``red?`` reads as a question and
        ``?3`` hides which field is known — and models misread it."""
        out = []
        for i, k in enumerate(state.knowledge[seat]):
            col = k["color"] if k["color"] else "unknown"
            rnk = str(k["rank"]) if k["rank"] else "unknown"
            parts = [f"colour {col}", f"rank {rnk}"]
            if k["not_colors"]:
                parts.append("not " + "/".join(k["not_colors"]))
            if k["not_ranks"]:
                parts.append("not rank " + "/".join(str(r) for r in k["not_ranks"]))
            out.append(f"[{i + 1}] " + ", ".join(parts))
        return "; ".join(out) if out else "(empty hand)"

    def _legal_clue_text(self, state: HanabiState, player: int) -> str:
        """Enumerate the clues that are actually legal right now.

        The action grammar is tiny but its LEGALITY space is not obvious: a
        clue must touch a card the recipient actually holds. Left to guess,
        models repeatedly proposed empty clues and burned the whole retry
        budget on them. Showing the legal set makes that failure impossible.
        """
        lines = []
        for other in range(self.n_players):
            if other == player:
                continue
            hand = state.hands[other]
            vals = [c for c in sorted({card["color"] for card in hand})]
            vals += [str(r) for r in sorted({card["rank"] for card in hand})]
            lines.append(f"    to P{other}: " + ", ".join(vals))
        return "\n".join(lines)

    # --------------------------------------------------------------- parsing
    _PLAY_RE = re.compile(r"<play>\s*(\d+)\s*</play>", re.I)
    _DISCARD_RE = re.compile(r"<discard>\s*(\d+)\s*</discard>", re.I)
    _CLUE_RE = re.compile(r"<clue(?:\s+to\s*=\s*P?(\d+))?\s*>\s*(.*?)\s*</clue>",
                          re.I | re.S)

    def parse_action(self, state: HanabiState, player: int, text: str) -> Action:
        hand_n = len(state.hands[player])

        m = self._PLAY_RE.search(text)
        if m:
            idx = int(m.group(1)) - 1
            if not 0 <= idx < hand_n:
                raise ParseError(f"play slot must be 1..{hand_n}")
            return {"type": "play", "index": idx}

        m = self._DISCARD_RE.search(text)
        if m:
            if state.clue_tokens >= MAX_CLUES:
                raise ParseError("cannot discard at 8 clue tokens; play or clue")
            idx = int(m.group(1)) - 1
            if not 0 <= idx < hand_n:
                raise ParseError(f"discard slot must be 1..{hand_n}")
            return {"type": "discard", "index": idx}

        m = self._CLUE_RE.search(text)
        if m:
            if state.clue_tokens <= 0:
                raise ParseError("no clue tokens left; play or discard")
            body = (m.group(2) or "").strip()
            to = m.group(1)
            if to is None:
                mt = re.search(r"\bP?(\d+)\b", body)
                if not mt:
                    raise ParseError("clue must name a player, e.g. <clue>P2 red</clue>")
                to = mt.group(1)
                body = body[:mt.start()] + body[mt.start():].replace(mt.group(0), "", 1)
            to = int(to)
            if to == player:
                raise ParseError("you cannot clue yourself")
            if not 0 <= to < self.n_players:
                raise ParseError(f"no such player P{to}")
            kind, value = self._parse_clue_value(body)
            touched = self._touched(state, to, kind, value)
            if not touched:
                raise ParseError(
                    f"empty clue: P{to} has no {value} card; a clue must touch "
                    "at least one card")
            return {"type": "clue", "to": to, "kind": kind, "value": value}

        raise ParseError("expected <play>N</play>, <discard>N</discard>, "
                         "or <clue>Pk colour|rank</clue>")

    def _parse_clue_value(self, body: str):
        low = body.lower()
        for c in self.colors:
            if re.search(rf"\b{c}\b", low):
                return "color", c
        m = re.search(r"\b([1-5])\b", low)
        if m:
            return "rank", int(m.group(1))
        # single-letter colour abbreviation, e.g. "P2 r"
        m = re.search(r"\b([rygbw])\b", low)
        if m and COLOR_ABBR[m.group(1)] in self.colors:
            return "color", COLOR_ABBR[m.group(1)]
        raise ParseError("clue value must be a colour "
                         f"({'/'.join(self.colors)}) or a rank 1-5")

    @staticmethod
    def _touched(state: HanabiState, seat: int, kind: str, value) -> List[int]:
        key = "color" if kind == "color" else "rank"
        return [i for i, c in enumerate(state.hands[seat]) if c[key] == value]

    # ------------------------------------------------------------------ step
    def step(self, state: HanabiState, action: Action) -> HanabiState:
        t = action.get("type")
        actor = state.to_move

        if t == "advance_phase":
            return state
        if t == "play":
            self._do_play(state, actor, int(action["index"]))
        elif t == "discard":
            self._do_discard(state, actor, int(action["index"]))
        elif t == "clue":
            self._do_clue(state, actor, action)
        else:
            return state

        state.turn += 1
        # One final round once the deck runs dry.
        if state.final_turns_left is not None:
            state.final_turns_left -= 1
        elif not state.deck:
            state.final_turns_left = self.n_players
        state.to_move = (actor + 1) % self.n_players
        self._maybe_finish(state)
        return state

    def _draw(self, state: HanabiState, seat: int) -> Optional[dict]:
        if not state.deck:
            return None
        card = state.deck.pop()
        state.hands[seat].append(card)
        state.knowledge[seat].append(_fresh_knowledge())
        return card

    def _do_play(self, state: HanabiState, seat: int, idx: int) -> None:
        card = state.hands[seat].pop(idx)
        state.knowledge[seat].pop(idx)
        color, rank = card["color"], card["rank"]
        ok = state.fireworks[color] == rank - 1
        if ok:
            state.fireworks[color] = rank
            if rank == MAX_RANK and state.clue_tokens < MAX_CLUES:
                state.clue_tokens += 1
        else:
            state.discards.append(card)
            state.fuse_tokens -= 1
        drew = self._draw(state, seat)
        state.last_event = {
            "kind": "play", "player": seat, "slot": idx, "card": dict(card),
            "success": ok, "fuses_left": state.fuse_tokens,
            "score": state.fireworks_score, "_drew": dict(drew) if drew else None,
        }

    def _do_discard(self, state: HanabiState, seat: int, idx: int) -> None:
        card = state.hands[seat].pop(idx)
        state.knowledge[seat].pop(idx)
        state.discards.append(card)
        if state.clue_tokens < MAX_CLUES:
            state.clue_tokens += 1
        drew = self._draw(state, seat)
        state.last_event = {
            "kind": "discard", "player": seat, "slot": idx, "card": dict(card),
            "clue_tokens": state.clue_tokens,
            "_drew": dict(drew) if drew else None,
        }

    def _do_clue(self, state: HanabiState, seat: int, action: Action) -> None:
        to = int(action["to"])
        kind = action["kind"]
        value = action["value"]
        key = "color" if kind == "color" else "rank"
        touched = self._touched(state, to, kind, value)
        state.clue_tokens -= 1
        for i, know in enumerate(state.knowledge[to]):
            if i in touched:
                know[key] = value
            else:
                bucket = "not_colors" if kind == "color" else "not_ranks"
                if value not in know[bucket]:
                    know[bucket].append(value)
        rec = {"turn": state.turn, "from": seat, "to": to, "kind": kind,
               "value": value, "touched": touched}
        state.clue_log.append(rec)
        state.last_event = {"kind": "clue", **rec,
                            "clue_tokens": state.clue_tokens}

    def _maybe_finish(self, state: HanabiState) -> None:
        if state.fuse_tokens <= 0:
            state.end_reason = "bombed"
        elif state.fireworks_score >= self.max_score:
            state.end_reason = "perfect"
        elif state.final_turns_left is not None and state.final_turns_left <= 0:
            state.end_reason = "deck_exhausted"
        else:
            return
        state.phase = PH_TERMINAL
        state.winner = "team"          # cooperative: one shared outcome
        state.win_reason = (f"score {self.score(state)}/{self.max_score} "
                            f"({state.end_reason}; fireworks "
                            f"{state.fireworks_score})")

    # ------------------------------------------------------- observations
    def observations(self, prev_state, new_state, action: Action,
                     actor: int) -> List[Obs]:
        """Hanabi is open information apart from own-hand identity: every
        action and its outcome is public. The card DRAWN to replace a played
        or discarded card must NOT reach its owner, so it is stripped from the
        delivered payload and kept only in the god log."""
        ev = getattr(new_state, "last_event", None)
        if not ev:
            return []
        public = {k: v for k, v in ev.items() if not k.startswith("_")}
        public["type"] = "hanabi_event"
        god = dict(public)
        if ev.get("_drew"):
            god["drew"] = ev["_drew"]
        return [Obs(audience=list(range(self.n_players)), payload=public,
                    log=god)]

    # ------------------------------------------------------------- terminal
    def is_terminal(self, state: HanabiState) -> bool:
        return state.phase == PH_TERMINAL

    def score(self, state: HanabiState) -> int:
        """Match score under the configured bomb convention."""
        if self.strict_bombs and state.fuse_tokens <= 0:
            return 0
        return state.fireworks_score

    def rewards(self, state: HanabiState) -> List[float]:
        """One shared payoff, identical for every seat (normalised 0..1)."""
        r = self.score(state) / float(self.max_score)
        return [r] * self.n_players

    # ------------------------------------------------------------ watcher
    def god_view(self, state: HanabiState) -> dict:
        return {
            "hands": [[dict(c) for c in h] for h in state.hands],
            "deck_remaining": len(state.deck),
            "deck_top": [dict(c) for c in state.deck[-5:]],
        }

    def snapshot(self, state: HanabiState) -> dict:
        return {
            "fireworks": dict(state.fireworks),
            "fireworks_score": state.fireworks_score,
            "score": self.score(state),
            "clue_tokens": state.clue_tokens,
            "fuse_tokens": state.fuse_tokens,
            "deck": len(state.deck),
            "discards": [_fmt(c) for c in state.discards],
            "to_move": state.to_move,
            "final_turns_left": state.final_turns_left,
            "end_reason": state.end_reason,
        }

    def render_board(self, state: HanabiState, *, reveal: str = "god") -> str:
        lines = [self._board_text(state)]
        for seat in range(self.n_players):
            if reveal == "god":
                cards = " ".join(_fmt(c) for c in state.hands[seat])
            else:
                cards = self._knowledge_line(state, seat)
            lines.append(f"  P{seat}: {cards}")
        return "\n".join(lines)
