"""Shared negotiation / messaging machinery (say / whisper / pass_talk).

Messaging is an ordinary Action type handled in a negotiation sub-phase via a
shared mixin — the runner loop is NOT special-cased (spec §0). A game keeps a
``NegotiationState`` on its state object (attribute ``nego``) and delegates its
own ``active_player`` / ``legal_actions`` / ``parse_action`` / ``step`` /
``observations`` to the ``nego_*`` helpers while ``state.phase`` is the
negotiation phase.

Masking is enforced in exactly two places (INV-2):
  * ``render_message_log`` (the transcript the game splices into
    ``render_prompt``); and
  * ``nego_observations`` (the per-action routing, via ``Obs`` audiences).

Regex here parses message *structure* only (tags), never quality (project
rule: story/message quality is judged by an LLM, never regex).

See spec §3.2 / §8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .base import GOD, Action, Obs, ParseError

MAX_MSG_CHARS_DEFAULT = 600


@dataclass
class NegotiationState:
    active: bool = False
    round_idx: int = 0
    max_rounds: int = 2
    speak_queue: List[int] = field(default_factory=list)   # seats yet to speak this round
    budget: Dict[int, int] = field(default_factory=dict)   # seat -> messages left this slot
    transcript: List[dict] = field(default_factory=list)   # god-view message records
    return_phase: str = ""                                 # phase to resume when negotiation ends


# Structure-only parse tags.
_SAY_RE = re.compile(r"<say>(.*?)</say>", re.IGNORECASE | re.DOTALL)
_WHISPER_RE = re.compile(
    r"<whisper\s+to\s*=\s*([0-9,\s]+)>(.*?)</whisper>",
    re.IGNORECASE | re.DOTALL,
)
_PASS_RE = re.compile(r"<pass\s*/?>(?:</pass>)?", re.IGNORECASE)


def _parse_seat_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.replace(" ", "").split(","):
        if tok == "":
            continue
        out.append(int(tok))
    return out


class MessagingMixin:
    """Reusable negotiation behaviour. Composed as a base class by games that
    keep a ``NegotiationState`` on their state object under attribute ``nego``.
    """

    # ----------------------------------------------------------- lifecycle
    def start_negotiation(self, state, *, return_phase: str,
                          rounds: "Optional[int]" = None) -> None:
        """Open a negotiation sub-phase; resume ``return_phase`` when done."""
        nego: NegotiationState = getattr(state, "nego")
        cfg = getattr(self, "config", None)
        default_rounds = getattr(cfg, "nego_rounds", 2) if cfg else 2
        nego.active = True
        nego.round_idx = 0
        nego.max_rounds = rounds if rounds is not None else default_rounds
        nego.return_phase = return_phase
        self._refill_round(state)

    def living_seats(self, state) -> List[int]:
        """Default: every seat. Games with eliminations override this."""
        return list(range(getattr(self, "n_players", 0)))

    def nego_round_order(self, state) -> List[int]:
        """Speaking order for one negotiation round. Default: living seats."""
        return list(self.living_seats(state))

    def _refill_round(self, state) -> None:
        """Rebuild ``speak_queue`` + per-seat ``budget`` for a new round."""
        nego: NegotiationState = getattr(state, "nego")
        cfg = getattr(self, "config", None)
        per_slot = getattr(cfg, "msgs_per_slot", 2) if cfg else 2
        order = self.nego_round_order(state)
        nego.speak_queue = list(order)
        nego.budget = {seat: int(per_slot) for seat in order}

    def _exit_negotiation(self, state) -> None:
        """Close negotiation and resume the saved return phase."""
        nego: NegotiationState = getattr(state, "nego")
        nego.active = False
        nego.speak_queue = []
        nego.budget = {}
        state.phase = nego.return_phase

    # ----------------------------------------------- runner-facing helpers
    def nego_active_player(self, state) -> int:
        nego: NegotiationState = getattr(state, "nego")
        if not nego.active or not nego.speak_queue:
            return -1
        return nego.speak_queue[0]

    def nego_legal_actions(self, state, player: int) -> List[Action]:
        """say (public), pass_talk, whisper to each non-empty living subset
        is large; we enumerate single-recipient whispers + say + pass, plus
        any alliance actions if this game composes ``AllianceMixin``.
        """
        cfg = getattr(self, "config", None)
        actions: List[Action] = [{"type": "pass_talk"}]
        if getattr(cfg, "messaging", True):
            actions.append({"type": "say", "text": ""})
            for other in self.living_seats(state):
                if other != player:
                    actions.append({"type": "whisper", "to": [other], "text": ""})
        if getattr(cfg, "alliances", True) and hasattr(self, "alliance_legal_actions"):
            actions.extend(self.alliance_legal_actions(state, player))
        return actions

    def nego_parse(self, state, player: int, text: str) -> Action:
        """Parse a negotiation response. Tries alliance tags first (delegated
        to AllianceMixin if present), then say/whisper/pass."""
        # Alliance tags take precedence (a turn may carry one alliance action).
        if hasattr(self, "alliance_parse"):
            try:
                return self.alliance_parse(state, player, text)
            except ParseError:
                pass
        m = _WHISPER_RE.search(text)
        if m:
            to = _parse_seat_list(m.group(1))
            body = m.group(2)
            living = set(self.living_seats(state))
            cleaned: List[int] = []
            for seat in to:
                if seat == player:
                    # Models routinely address a group by listing every seat,
                    # including their own ("to=1,2,3,4" from P3). The intent is
                    # unambiguous, so drop the sender rather than void the match;
                    # a whisper addressed ONLY to self still fails the
                    # at-least-one-recipient check below.
                    continue
                if seat not in living:
                    raise ParseError(f"whisper recipient {seat} is not living")
                if seat not in cleaned:
                    cleaned.append(seat)
            if not cleaned:
                raise ParseError("whisper requires at least one recipient")
            return {"type": "whisper", "to": sorted(cleaned), "text": body.strip()}
        m = _SAY_RE.search(text)
        if m:
            return {"type": "say", "text": m.group(1).strip()}
        if _PASS_RE.search(text):
            return {"type": "pass_talk"}
        raise ParseError("expected <say>...</say>, <whisper to=..>..</whisper>, "
                         "or <pass>")

    def _truncate(self, text: str):
        """Return (text, truncated_flag) honouring config.max_msg_chars."""
        cfg = getattr(self, "config", None)
        cap = getattr(cfg, "max_msg_chars", MAX_MSG_CHARS_DEFAULT) if cfg \
            else MAX_MSG_CHARS_DEFAULT
        if len(text) > cap:
            return text[:cap], True
        return text, False

    def nego_step(self, state, action: Action):
        """Record a message, decrement budget, advance the speak queue.

        For an alliance action, mutation is delegated to
        ``apply_alliance_action`` (the message bookkeeping below still
        consumes one slot so alliance dealings are paced like talk).
        """
        nego: NegotiationState = getattr(state, "nego")
        t = action.get("type")
        speaker = nego.speak_queue[0] if nego.speak_queue else -1
        turn = getattr(state, "turn", 0)

        if t in ("say", "whisper"):
            text, truncated = self._truncate(str(action.get("text", "")))
            action["text"] = text
            if truncated:
                action["truncated"] = True
            entry = {
                "type": "message",
                "scope": "public" if t == "say" else "private",
                "from": speaker,
                "text": text,
                "turn": turn,
            }
            if t == "whisper":
                entry["to"] = list(action.get("to", []))
            nego.transcript.append(entry)
        elif t.startswith("alliance_") and hasattr(self, "apply_alliance_action"):
            self.apply_alliance_action(state, speaker, action)
        # pass_talk: nothing recorded.

        # Budget / queue bookkeeping.
        if speaker >= 0:
            nego.budget[speaker] = nego.budget.get(speaker, 0) - 1
            # Pop the current speaker.
            nego.speak_queue = nego.speak_queue[1:]
            if t == "pass_talk":
                # Passing forfeits the rest of this slot.
                nego.budget[speaker] = 0
            elif nego.budget[speaker] > 0:
                # Re-queue at the end for fair interleave.
                nego.speak_queue.append(speaker)

        # Round / phase advancement.
        if not nego.speak_queue:
            nego.round_idx += 1
            if nego.round_idx >= nego.max_rounds:
                self._exit_negotiation(state)
            else:
                self._refill_round(state)
        return state

    def nego_observations(self, prev_state, new_state, action: Action,
                          actor: int) -> List[Obs]:
        """Route + mask a just-applied negotiation action."""
        t = action.get("type")
        living = list(self.living_seats(new_state))

        if t == "say":
            payload = {"type": "message", "scope": "public",
                       "from": actor, "text": action.get("text", "")}
            return [Obs(audience=living, payload=payload)]

        if t == "whisper":
            to = list(action.get("to", []))
            recipients = sorted(set(to) | {actor})
            full = {"type": "message", "scope": "private", "from": actor,
                    "to": list(to), "text": action.get("text", "")}
            # Recipients (+ sender) get full text; god-log carries full text.
            obs_list = [Obs(audience=recipients, payload=full)]
            cfg = getattr(self, "config", None)
            visibility = getattr(cfg, "whisper_visibility", "metadata") if cfg \
                else "metadata"
            if visibility == "metadata":
                bystanders = sorted(set(living) - set(recipients))
                if bystanders:
                    meta = {"type": "message_meta", "from": actor,
                            "n_recipients": len(to)}
                    # Bystanders see only metadata; the god-log still has full
                    # text (logged via the `log=` channel below).
                    obs_list.append(Obs(audience=bystanders, payload=meta,
                                        log=dict(full)))
            return obs_list

        if t and t.startswith("alliance_") and hasattr(self, "alliance_observations"):
            return self.alliance_observations(new_state, action, actor)

        # pass_talk and anything else: no observation.
        return []

    # ------------------------------------------------------------- masking
    def render_message_log(self, state, player: int) -> str:
        """Return the MASKED transcript for ``player`` (the single masking
        point for the transcript inside ``render_prompt``). A seat sees public
        messages, whispers it sent, and whispers it received — nothing else.
        """
        nego: NegotiationState = getattr(state, "nego", None)
        if nego is None or not nego.transcript:
            return ""
        lines: List[str] = []
        for e in nego.transcript:
            scope = e.get("scope")
            frm = e.get("from")
            if scope == "public":
                lines.append(f"[PUBLIC] P{frm}: {e.get('text','')}")
            else:
                to = e.get("to", [])
                if player == frm or player in to:
                    tag = ",".join(f"P{x}" for x in to)
                    lines.append(f"[WHISPER P{frm}->{tag}] {e.get('text','')}")
        return "\n".join(lines)
