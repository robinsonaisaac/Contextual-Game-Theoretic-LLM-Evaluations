# FINAL ENGINEERING SPEC — Messaging, Alliances, Observability for `game_theory_llm/play/`

Lead-architect synthesis of Designs A/B/C. This is the authoritative document; downstream agents implement strictly from it. Where the three designs conflicted, the decision and one-line rationale are stated inline as **[DECISION]**.

Repo root for all paths: `/Users/isaacrobinson/Documents/Contextual-Game-Theoretic-LLM-Evaluations/.worktrees/steering/`. The package is `game_theory_llm/play/`.

---

## 0. Cross-cutting decisions (read first)

- **[DECISION] Observation hook signature = `observations(self, prev_state, new_state, action, actor) -> list[Observation]`.** Rationale: Design B's two-state signature is the only one that lets a game describe *results* (combat dice, vote tallies, reveals) without re-deriving them; A's single-state and C's single-state cannot. Backward-compat default ignores `prev_state`.
- **[DECISION] `Observation` is a small dataclass, not a bare tuple.** Rationale: B's `Observation(audience, payload, log)` separates the masked per-player payload from the full god-view log payload cleanly; C's `(audience, obs)` tuples cannot carry a distinct god-log. Audience is always a concrete `list[int]`; the pseudo-seat `GOD = -1` may appear only inside the *logged* audience, never delivered to a player.
- **[DECISION] Messaging/alliances are ordinary Action types handled in a negotiation sub-phase via a shared mixin.** All three designs agreed. The runner loop is NOT special-cased.
- **[DECISION] Action type namespace: `say` / `whisper` / `pass_talk` for messages; `alliance_propose` / `alliance_accept` / `alliance_decline` / `alliance_break` for alliances.** Rationale: A's verbs are the most self-documenting. (B's `message`+`scope` and C's `msg_public`/`msg_private` are rejected as either overloaded or verbose.)
- **[DECISION] Alliances are cheap-talk by default (no hard enforcement); Risk gets an *optional* `enforce_alliances` toggle.** Rationale: all three agreed — we must be able to *measure* betrayal, so betrayal must remain physically possible. The toggle lets us A/B "can't betray" vs "chooses not to".
- **[DECISION] Steering instrumentation (per-record `steering` tag, `alliance_summary`, derived-metric design) from Design C is adopted wholesale.** Rationale: this harness exists to measure steering; C is the only design that made the metrics a pure log reduction. Message scoring uses an LLM judge (Sonnet+), never regex (project rule).
- **[DECISION] Watcher is a pure log reader (`viewer.py`) with no game-engine import, EXCEPT per-game board art via a `render_board` registry.** Rationale: B/C agree the viewer must prove the log is self-sufficient; board art is the one game-specific concession and lives in each game file so the viewer stays game-agnostic.
- **[DECISION] State is mutated in place by every existing game (`new = state`).** Therefore the runner MUST capture `prev_phase` (and a shallow snapshot for `phase_change`) BEFORE calling `step`. This is load-bearing and called out in §2.
- **[DECISION] Diplomacy ships "standard-map Full-Press with documented simplifications" (no multi-fleet convoy chains, Szykman-rule paradox handling, single-province split coasts), gated behind a pure `adjudicate()` function tested against a DATC subset.** Rationale: all three flagged full DATC as multi-week/unbounded risk; this bounds it honestly. See §5 and §9.

Backward-compatibility invariants (must hold; enforced by tests):
- **INV-1**: A game that overrides nothing new behaves byte-identically to today (broadcast every action to all players).
- **INV-2**: All masking lives in `Game.render_prompt` + `Game.observations`. The runner only routes.
- **INV-3**: v1 JSONL logs replay in the watcher unchanged; every v1 record keeps its shape; v2 adds fields/types only.
- **INV-4**: Existing tests stay green unmodified.

---

## 1. Final `base.py` changes (exact signatures, backward-compatible)

All additions are concrete (non-abstract) methods with defaults, so existing games and `Game` subclasses keep working untouched. Append the following to `game_theory_llm/play/base.py`:

```python
# ---- NEW module-level symbols ----
from dataclasses import dataclass, field
from typing import Any, List, Optional

GOD = -1  # pseudo-seat: appears only in the *logged* audience, never delivered to a player.

Observation = dict  # alias kept for readability; payloads are JSON-serialisable dicts with a "type" key.

@dataclass
class Obs:
    """One routed observation produced by `Game.observations`.

    audience : concrete seat indices that receive `payload` via receive_observation.
               NEVER contains GOD(-1). [] == delivered to nobody (god-log only).
    payload  : the MASKED dict handed to each audience member's receive_observation.
    log      : the FULL god-view dict written to the JSONL `observation` record.
               If None, the runner logs `payload`. Use this to log full whisper
               text while delivering a redacted payload to bystanders.
    """
    audience: List[int]
    payload: dict
    log: Optional[dict] = None
```

Add these concrete methods to `class Game`:

```python
    def observations(self, prev_state, new_state, action: "Action", actor: int) -> List["Obs"]:
        """Return observations generated by `actor` taking `action`.

        Called by the runner AFTER step(). `prev_state` is the pre-step state,
        `new_state` is the post-step state (note: existing games mutate in place,
        so prev_state and new_state may be the SAME object — see §2).

        DEFAULT (backward compatible, == legacy runner): broadcast a public
        'action' obs to every seat. A game that does not override this behaves
        exactly as today.
        """
        return [Obs(audience=list(range(self.n_players)),
                    payload={"type": "action", "player": actor, "action": action})]

    def god_view(self, state) -> dict:
        """Optional. Return god-view hidden truth for the watcher (roles, deck top,
        center cards, unit ownership). Written once to a `setup` record after
        initial_state and refreshed in each `state_snapshot`. Default {}."""
        return {}

    def snapshot(self, state) -> dict:
        """Optional. Public + hidden board snapshot for the watcher's state_snapshot
        records (emitted at match start and on every phase change). Default: a
        best-effort JSON-safe dump (runner falls back to its own _safe_dump)."""
        return {}

    def render_board(self, state, *, reveal: str = "god") -> str:
        """Optional. ASCII board art for the watcher. `reveal` in {"god","public"}
        or "seatN". Default '' (viewer shows the structured log only)."""
        return ""
```

Extend the `Player` Protocol docstring only (no signature change): `receive_observation(obs)` now may receive new `obs["type"]` values `"message"`, `"message_meta"`, and `"alliance_event"` in addition to `"action"`, `"parse_error"`, `"phase_change"`.

**No abstract method is added or changed.** `Action`, `ParseError`, `MatchResult`, all seven abstract `Game` methods, and `Player` keep their exact current signatures.

---

## 2. Final `runner.py` changes (message routing / private observation delivery)

Replace the broadcast block (current lines 130–139) and augment logging. The complete changed `run_match` body, in diff-intent form:

**2a. `match_start` record** — add `schema`, `config`, `steering_tags`, and write a `setup` record + initial `state_snapshot`:

```python
    cfg = getattr(game, "config", None)
    log({
        "type": "match_start", "schema": 2,
        "game": game.name, "n_players": game.n_players, "seed": seed,
        "players": [getattr(p, "name", type(p).__name__) for p in players],
        "config": (cfg.as_dict() if hasattr(cfg, "as_dict") else {}),
        "steering_tags": [getattr(p, "steering_tag", None) for p in players],
    })
    log({"type": "setup", "turn": 0, "god_view": game.god_view(state)})
    log({"type": "state_snapshot", "turn": 0,
         "phase": getattr(state, "phase", None), "snapshot": game.snapshot(state)})
```

**2b. Stamp `event_id`, `match_id`, and `steering` on every record.** `log()` gains a monotonic counter:

```python
    import uuid
    match_id = uuid.uuid4().hex
    _eid = {"n": 0}
    def log(rec: dict) -> None:
        rec.setdefault("ts", _now())
        rec["event_id"] = _eid["n"]; _eid["n"] += 1
        rec["match_id"] = match_id
        with log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
```

The `prompt` and `action` records additionally carry `"steering": getattr(players[active], "steering_tag", None)` and `"phase": getattr(state, "phase", None)`.

**2c. Capture phase BEFORE step (in-place mutation safety — load-bearing):**

```python
        prev_phase = getattr(state, "phase", None)
        try:
            new_state = game.step(state, action)
        except Exception as e:
            log({"type": "step_error", "turn": turn, "error": str(e), "action": action})
            break
        state = new_state
        post_phase = getattr(state, "phase", None)
        if post_phase != prev_phase:
            log({"type": "phase_change", "turn": turn, "from": prev_phase,
                 "to": post_phase, "active_seat": game.active_player(state)})
            log({"type": "state_snapshot", "turn": turn, "phase": post_phase,
                 "snapshot": game.snapshot(state)})
            # Notify chat players of phase change (existing llm.py already handles it).
            for p in players:
                try: p.receive_observation({"type": "phase_change", "to": post_phase})
                except Exception: pass
```

**2d. Replace the broadcast with the routing courier:**

```python
        try:
            obs_list = game.observations(state, state, action, actor=active)
        except Exception:
            obs_list = [Obs(audience=list(range(game.n_players)),
                            payload={"type": "action", "player": active, "action": action})]
        for o in obs_list:
            logged = dict(o.log if o.log is not None else o.payload)
            logged.setdefault("turn", turn)
            log({"type": "observation", "turn": turn, "actor": active,
                 "audience": list(o.audience), "obs": logged})
            for seat in o.audience:
                if seat < 0:          # GOD/spectator pseudo-seat: never delivered
                    continue
                try:
                    payload = dict(o.payload); payload.setdefault("turn", turn)
                    players[seat].receive_observation(payload)
                except Exception:
                    pass
```

> Note on `prev_state`: existing games mutate in place, so we pass `state` for both args (they reference the same post-step object). Games that need true pre/post diffs must snapshot internally; the default and all four games here describe results from `new_state` + `action`, which is sufficient.

**2e. `terminal` record** — add `winner`, `win_reason`, `alliance_summary`:

```python
    log({"type": "terminal", "turn": turn, "rewards": rewards,
         "state": _safe_dump(state),
         "winner": getattr(state, "winner", getattr(state, "winner_team", None)),
         "win_reason": getattr(state, "win_reason", ""),
         "alliance_summary": _alliance_summary_or_empty(state)})
```

where `_alliance_summary_or_empty(state)` calls `alliances.alliance_summary(state.alli)` if the state has an `alli` field, else `{}`.

The `advance_phase` branch, parse-retry loop, `fallback`, and `noop` fallback are unchanged. Import `from .base import Obs, GOD` and `from .alliances import alliance_summary`.

---

## 3. Shared messaging + alliance modules

### 3.1 `game_theory_llm/play/config.py` (NEW)

```python
@dataclass
class GameConfig:
    messaging: bool = True
    alliances: bool = True
    nego_rounds: int = 2          # negotiation rounds per negotiation phase (game may override)
    msgs_per_slot: int = 2        # messages a seat may send per speaking slot
    max_msg_chars: int = 600
    enforce_alliances: bool = False   # Risk only: if True, alliance-violating attacks illegal
    whisper_visibility: str = "metadata"   # "metadata" -> bystanders learn a whisper happened; "hidden" -> nothing
    def as_dict(self) -> dict: ...   # for match_start.config
```

Each `Game.__init__` accepts `config: GameConfig | None = None` and stores `self.config = config or GameConfig()`. (This is the one additive `__init__` change per game; the no-arg constructor still works → INV-4.)

### 3.2 `game_theory_llm/play/messaging.py` (NEW)

```python
MAX_MSG_CHARS_DEFAULT = 600

@dataclass
class NegotiationState:
    active: bool = False
    round_idx: int = 0
    max_rounds: int = 2
    speak_queue: list[int] = field(default_factory=list)   # seats yet to speak this round
    budget: dict[int, int] = field(default_factory=dict)   # seat -> messages left this slot
    transcript: list[dict] = field(default_factory=list)   # god-view message records
    return_phase: str = ""                                 # phase to resume when negotiation ends

class MessagingMixin:
    # ---- lifecycle ----
    def start_negotiation(self, state, *, return_phase: str, rounds: int | None = None) -> None
    def living_seats(self, state) -> list[int]                # default: range(n_players); games override
    def nego_round_order(self, state) -> list[int]            # default: living_seats; games override
    def _refill_round(self, state) -> None                    # rebuild speak_queue + budgets
    def _exit_negotiation(self, state) -> None                # set state.phase = nego.return_phase
    # ---- runner-facing helpers (delegated to from game's own methods) ----
    def nego_active_player(self, state) -> int                # speak_queue[0] or -1
    def nego_legal_actions(self, state, player) -> list[Action]
    def nego_parse(self, state, player, text) -> Action       # parses <say>/<whisper>/<pass>/<ally...>
    def nego_step(self, state, action) -> Any                 # records msg, decrements budget, advances queue
    def nego_observations(self, prev, new, action, actor) -> list[Obs]
    def render_message_log(self, state, player) -> str        # MASKED transcript for render_prompt
```

**Message Action dict schemas** (JSON-serialisable, `type` key required):

```python
{"type": "say",       "text": str}                      # PUBLIC: all living seats + god
{"type": "whisper",   "to": [int, ...], "text": str}    # PRIVATE: to ∪ {sender} + god(full)
{"type": "pass_talk"}                                   # spend a slot saying nothing
```

Parse tags (`nego_parse`, regex-based parse of *structure* only — never quality):
- `<say>…</say>` → `say`
- `<whisper to=2,4>…</whisper>` → `whisper` (comma seat list)
- `<pass></pass>` → `pass_talk`
- alliance tags delegate to `AllianceMixin.alliance_parse` (§3.3).

Constraints enforced in `nego_parse`/`nego_legal_actions`:
- `to` non-empty, living, deduped, excludes sender; `ParseError` otherwise.
- `text` truncated to `config.max_msg_chars` with `truncated: true` flag (never rejected — a verbose model must not stall a match).
- A seat may emit ≤ `budget[seat]` messages before its slot ends; `pass_talk` or budget exhaustion pops it from `speak_queue` and (if budget remains and game allows) re-queues at the end for fair interleave.

`nego_observations` routing/masking:
- `say` → one `Obs(audience=living, payload={"type":"message","scope":"public","from":actor,"text":...})`.
- `whisper` → `Obs(audience=sorted(to∪{actor}), payload=full)`; if `config.whisper_visibility=="metadata"`, a second `Obs(audience=bystanders, payload={"type":"message_meta","from":actor,"n_recipients":len(to)})`; the god-log payload (`log=`) always carries full text via `audience=[GOD]` semantics (the routing record's logged `obs` has full text regardless).
- `pass_talk` → `[]`.

`render_message_log(state, player)` returns only transcript entries where `scope=="public" or player in entry["to"] or player==entry["from"]`. **This is the single masking point for the transcript inside `render_prompt`.**

### 3.3 `game_theory_llm/play/alliances.py` (NEW)

```python
@dataclass
class Alliance:
    id: int
    members: list[int]                 # sorted seats (>=2)
    proposer: int
    kind: str                          # per-game enum (see §5)
    terms: dict                        # freeform, game-specific (logged; enforced only where noted)
    status: str                        # "proposed" | "active" | "declined" | "broken" | "expired" | "honored"
    proposed_turn: int
    accepted_turn: int | None = None
    broken_turn: int | None = None
    broken_by: int | None = None
    pending: list[int] = field(default_factory=list)   # invitees who haven't accepted

@dataclass
class AllianceState:
    next_id: int = 0
    alliances: dict[int, Alliance] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)   # full audit trail (mirrors JSONL alliance_event)

class AllianceMixin:
    def alliance_legal_actions(self, state, player) -> list[Action]
    def alliance_parse(self, state, player, text) -> Action          # <ally ...> tags
    def apply_alliance_action(self, state, actor, action) -> list[dict]   # mutates AllianceState; returns event records
    def alliance_observations(self, state, action, actor) -> list[Obs]
    def alliance_effects(self, state) -> dict                        # per-game; default {}
    def judge_alliance(self, state, action, actor) -> list[dict]     # per-game; emit honored/betrayed events at board resolution
    def is_allied(self, state, a: int, b: int) -> bool               # active alliance membership test

# module-level
def alliance_summary(alli: AllianceState) -> dict   # the terminal-record reduction (see §4)
def new_event(...) -> dict                          # builds a canonical alliance_event dict
```

**Alliance Action dict schemas:**

```python
{"type": "alliance_propose", "to": [int, ...], "kind": str, "terms": dict}
{"type": "alliance_accept",  "alliance_id": int}
{"type": "alliance_decline", "alliance_id": int}
{"type": "alliance_break",   "alliance_id": int, "reason": str}     # explicit defection
```

Parse tags: `<ally propose to=2,3 kind=nonaggression>terms text</ally>`, `<ally accept 7>`, `<ally decline 7>`, `<ally break 7>reason</ally>`.

Lifecycle: `propose` creates an `Alliance(status="proposed", pending=to)`; an alliance becomes `active` only when **all** `pending` have `accept`ed (partial accepts recorded, inert). `decline` by any invitee sets `status="declined"`. `break` sets `status="broken"`, `broken_by=actor`. Honour/betray are emitted by `judge_alliance` at the per-game board-action resolution point (not by the alliance action itself).

Routing (`alliance_observations`): `propose`/`accept`/`decline` route to `members ∪ pending` (+ full god-log); `break` routes to all affected members AND is publicly broadcast (a betrayal is observable to its victims by definition).

---

## 4. Log schema v2 — every event type with exact fields

JSONL, one record per line, append-only. **Every record carries `ts` (str), `event_id` (monotonic int from 0), `match_id` (str).** `match_start.schema == 2`. v1 records keep their shape (INV-3); new types/fields are additive.

| `type` | Exact fields (beyond `ts`/`event_id`/`match_id`) | Notes |
|---|---|---|
| `match_start` | `schema:2`, `game:str`, `n_players:int`, `seed:int`, `players:[str]`, `config:{}`, `steering_tags:[{vector,alpha,layer,model}\|null]` | `config` from `GameConfig.as_dict()`; `steering_tags[i]` = `players[i].steering_tag` or null |
| `setup` (NEW) | `turn:0`, `god_view:{}` | `Game.god_view(state)`: roles/center/deck-top/unit ownership. God-only; never in a prompt |
| `state_snapshot` (NEW) | `turn:int`, `phase:str\|null`, `snapshot:{public:{}, hidden:{}}` | Emitted at start + every `phase_change`. Watcher reconstructs board from these |
| `phase_change` (NEW) | `turn:int`, `from:str\|null`, `to:str\|null`, `active_seat:int` | Runner-emitted phase diff |
| `prompt` | `turn:int`, `player:int`, `prompt:str`, `n_legal:int`, `phase:str` (NEW), `steering:{}\|null` (NEW) | v1 fields unchanged; two additive |
| `action` | `turn:int`, `player:int`, `action:{}`, `phase:str` (NEW), `steering:{}\|null` (NEW), `truncated:bool` (NEW, msgs only) | v1 unchanged + additive |
| `parse_error` | `turn:int`, `player:int`, `error:str`, `retry:int` | unchanged |
| `fallback` | `turn:int`, `player:int`, `action:{}`, `reason:str` | unchanged |
| `observation` (NEW) | `turn:int`, `actor:int`, `audience:[int]` (may contain `-1`=GOD), `obs:{}` | The routing record. `obs.type` ∈ `action,message,message_meta,alliance_event,reveal,combat,resolution`. God-log always full content |
| `obs.message` (embedded) | `type:"message"`, `scope:"public"\|"private"`, `from:int`, `to:[int]?`, `text:str` | Full text in the logged `observation`; bystanders get `message_meta` |
| `obs.message_meta` (embedded) | `type:"message_meta"`, `from:int`, `n_recipients:int` | Content-free leak (config `whisper_visibility="metadata"`) |
| `alliance_event` (NEW) | `turn:int`, `event:"propose"\|"accept"\|"decline"\|"break"\|"honored"\|"betrayed"\|"expired"`, `alliance_id:int`, `kind:str`, `proposer:int`, `members:[int]`, `actor:int`, `counterparty:[int]`, `terms:{}`, `action_ref:{}\|null`, `alliance_age_turns:int\|null` | One per transition + each honour/betray judgement. `action_ref` = the board action that honoured/betrayed |
| `combat` (NEW) | `turn:int`, game-specific: Risk `{src,dst,atk_rolls,def_rolls,atk_lost,def_lost,captured:bool}`; Diplomacy `{target,incoming:[{src,strength}],defender_strength,outcome}` | For faithful replay of stochastic/resolution detail |
| `reveal` (NEW) | `turn:int`, `what:str`, `data:{}`, `audience:[int]` | Night peeks (ONW), Policy Peek / Investigate (SH); audience-scoped |
| `step_error` | `turn:int`, `error:str`, `action:{}` | unchanged |
| `advance_phase_error` | `error:str` | unchanged |
| `terminal` | `turn:int`, `rewards:[float]`, `state:{}`, `winner` (NEW), `win_reason:str` (NEW), `alliance_summary:{}` (NEW) | v1 + additive |

**`alliance_summary` (terminal) exact shape** — the offline-metrics reduction:

```json
{"n_proposed": int, "n_accepted": int, "n_declined": int, "n_broken": int,
 "n_honored": int, "n_betrayed": int,
 "per_player": {"<seat>": {"proposed": int, "accepted": int,
                           "honored": int, "betrayed": int, "betrayed_against": int}}}
```

**Reconstruction guarantee (tested):** the union of all `observation` records whose `audience` contains seat `s`, plus all `prompt` records for `s`, fully reconstructs seat `s`'s knowledge; the union of *everything* (incl. `setup`/`state_snapshot.hidden` and `-1`-audience full content) reconstructs god-view. `event_id` totally orders events within a turn.

**Derived steering metrics** (pure log reductions, in `metrics.py`): formation rate `n_accepted/n_proposed`; betrayal rate `n_betrayed/n_accepted`; honour rate `n_honored/n_accepted`; private-message ratio `count(message scope=private)/count(message)`; alliance survival = `alliance_age_turns` distribution at break/terminal; first-strike asymmetry = `turn` of first `betrayed` per steering arm. Message sentiment/aggression/deception scored post-hoc by an LLM judge (Sonnet+) into `msg_score` enrichment records keyed by the annotated `event_id` (**no regex**, per project rule).

`msg_score` (NEW, written only by `metrics.py`, never the runner): `{type:"msg_score", event_id:int (annotated), scored_by:str, sentiment:float, aggression:float, deception_flag:bool, rationale:str}`.

---

## 5. Per-game feature-parity checklist (faithful vs explicitly simplified)

### 5.1 One Night Werewolf — `play/games/one_night_werewolf.py`
**Faithfully implemented:**
- [ ] 3–10 players; deck size `n+3`; configurable role list via `GameConfig`/constructor arg `roles: list[str]|None`.
- [ ] Full role set: Werewolf, Minion, Mason×2, Seer, Robber, Troublemaker, Insomniac, Hunter, Tanner, Drunk, Doppelganger, Villager.
- [ ] Canonical wake order: Doppelganger → Werewolves → Minion → Masons → Seer → Robber → Troublemaker → Drunk → Insomniac. (Hunter/Tanner/Villager have no night action.)
- [ ] Doppelganger copies a target's role at wake; if the copied role is a night role, a dynamic second `night_queue` insertion lets the Doppelganger act in that role's slot.
- [ ] Minion (sees wolves, wins with wolves, death-safe), Masons (see each other), Insomniac (sees own final card post-swaps), Drunk (swaps with a center card blind), Robber/Troublemaker/Seer per canon.
- [ ] Win/reward edge cases: Tanner wins iff Tanner dies; Hunter on death eliminates their vote target; existing no-wolf and tie resolution preserved.
- [ ] DAY phase becomes the negotiation phase (public `say` + `whisper` + voting-bloc/truce alliances). VOTE phase unchanged but pact-instrumented.
**Explicitly simplified:** none required. (Doppelganger-as-Doppelganger and Doppelganger-Hunter-shoots-on-death are implemented; if timeline forces it, the *only* allowed cut is Doppelganger-Drunk re-swap nuance, which must be flagged in the module docstring + `config`.)

### 5.2 Secret Hitler — `play/games/secret_hitler.py`
**Faithfully implemented:**
- [ ] 5–10 players, official role counts per player count (5p:3L/1F/H … 10p:6L/3F/H per the standard table).
- [ ] Hitler-knowledge rule: 5–6p Fascists+Hitler know each other; 7–10p Hitler is blind (generalize `_private_header`).
- [ ] All executive powers on the correct per-player-count board schedule: Investigate Loyalty (President sees a target's party membership → `reveal` record, audience=president), Special Election (President picks next President), Policy Peek (President sees top 3 → `reveal`), Execution.
- [ ] Veto Power unlocked at 5 Fascist policies: Chancellor may propose veto in enact phase; President agrees → both discarded, election tracker +1; refuse → Chancellor must enact.
- [ ] Election tracker / chaos / reshuffle / term-limits preserved from v0.
- [ ] `PH_DISCUSSION` negotiation inserted before each `PH_NOMINATION`; `vote_pact`/`gov_pact`/`nonaggression` alliances; honour/betray judged at vote (`_resolve_vote`) and enact.
**Explicitly simplified:** none for the standard liberal/fascist track. The two *fascist* board variants (5–6p vs 7–8p vs 9–10p power schedules) are all implemented from the table; no rules cut.

### 5.3 Risk — `play/games/risk_lite.py` (keep filename; `name="risk"`)
**Faithfully implemented (data table in `play/maps/risk_map.py`):**
- [ ] 42-territory classic map, 6 continents + bonuses, full canonical adjacency (incl. sea bridges Alaska–Kamchatka, etc.).
- [ ] Reinforcements `max(3, territories//3)` + continent bonuses + Risk-card set trading (escalating set values `4,6,8,10,12,15` then `+5`).
- [ ] Attack-until-you-stop (`PH_ATTACK` stays active across multiple attacks until `end_attack`).
- [ ] Fortify phase (`PH_FORTIFY`: one move along an owned-connected path) then end turn.
- [ ] Elimination → attacker captures eliminated player's Risk cards; forced-trade at 5+ cards.
- [ ] Standard dice combat preserved; logged via `combat` records.
- [ ] `PH_NEGOTIATION` at start of each player's turn (before DEPLOY) + a pre-game round 0; `nonaggression`/`mutual_defense`/`dmz`/`coalition` alliances. `enforce_alliances` config: if False (default) attacking an ally is legal and auto-breaks the alliance (logged `betrayed`); if True, ally-violating attacks are filtered from `legal_actions` and rejected by `parse_action`.
**Explicitly simplified:** dice RNG remains deterministic-per-attack-seed (existing pattern) rather than drawing from the runner RNG — flagged in docstring; behaviourally faithful, just reproducible.

### 5.4 Diplomacy — `play/games/diplomacy_lite.py` (keep filename; `name="diplomacy"`) — **RISKIEST, see §9**
**Faithfully implemented (data table in `play/maps/diplomacy_map.py`):**
- [ ] Standard 7-power Europe, 34 supply centres, 75 provinces, inland/coastal/sea typing, home centres.
- [ ] Armies + fleets with type-legal moves (`ADJ_ARMY` / `ADJ_FLEET` graphs).
- [ ] Phase engine: Spring move → Spring retreat → Fall move → Fall retreat → Winter build/disband (`PH_SPRING/PH_RETREAT/PH_FALL/PH_WINTER`).
- [ ] Support (hold & move), support-cut by any non-supported attacker, standoff/bounce, self-dislodgement ban, dislodged-unit retreat-or-disband, builds/disbands from SC count on home centres.
- [ ] Full-Press `PH_NEGOTIATION` before each movement phase (≥2 rounds); `support_pact`/`dmz`/`nonaggression`/`coalition` alliances; honour = promised support order appears; betray = MOVE into ally's occupied/SC province, or omitting promised support.
- [ ] Adjudication isolated behind pure `adjudicate(orders, board) -> Resolution`; gated by `tests/test_game_diplomacy.py` DATC subset.
**Explicitly simplified (documented in docstring + `config.adjudicator="lite-press-v2"`, honestly flagged):**
- Convoys: single-fleet convoys only; **multi-fleet convoy chains disallowed** (`ParseError` at parse time).
- Convoy paradoxes (Pandin's/Betrayal): **Szykman rule** — paradoxical convoys treated as failed/hold. No fixpoint solver.
- Split coasts (Spain/St.P/Bulgaria): **modelled as single provinces** (no north/south-coast distinction).
- Retreats: deterministic lowest-index legal empty non-contested space, else disband. No retreat-standoff edge cases.
- Builds: on owned empty home SC only; excess builds waived.
- Skipped with documented deterministic resolution: beleaguered-garrison nuance, self-standoff subtleties.

---

## 6. Watch/replay tool spec — `play/viewer.py` + `scripts/watch_match.py`

**[DECISION] One module `play/viewer.py`, CLI `scripts/watch_match.py` (also runnable as `python3 -m game_theory_llm.play.viewer`).** Pure stdlib + JSONL; optional `rich` (degrade to plain ANSI if absent). No game-engine import except calling `Game.render_board` via a name→class registry built from `play/games`.

**Inputs / CLI:**
```
python3 scripts/watch_match.py LOG.jsonl                 # static replay, god-view, auto-paged
python3 scripts/watch_match.py LOG.jsonl --follow        # live-follow (tail -f, 200ms poll)
python3 scripts/watch_match.py LOG.jsonl --seat N        # reconstruct ONLY what seat N saw (masked)
python3 scripts/watch_match.py LOG.jsonl --god           # default: reveal all hidden info
python3 scripts/watch_match.py LOG.jsonl --step          # event-by-event stepper (n/p/q)
python3 scripts/watch_match.py LOG.jsonl --speed S       # auto-play S events/sec
python3 scripts/watch_match.py LOG.jsonl --html OUT.html # self-contained static HTML replay
python3 scripts/watch_match.py LOG.jsonl --metrics       # print steering metrics table (§4)
```

**Architecture:**
```python
def load_events(path) -> list[dict]                 # parse JSONL; skip un-terminated trailing line (for --follow)
def reconstruct(events) -> MatchReplay              # fold into ordered Frames
@dataclass
class Frame:
    event_id: int; turn: int; phase: str
    board: str                  # from Game.render_board(snapshot, reveal=...) — game art
    chats: list[dict]           # cumulative messages
    alliances: list[dict]       # current alliance ledger w/ status
    god_panel: str              # hidden truth (setup.god_view + state_snapshot.hidden)
    seat_panels: dict[int, str] # per-seat masked reconstruction
class MatchReplay:
    def render(self, idx, *, reveal: str = "god") -> str
class LogReader:                # tails file for --follow; yields parsed records
class TerminalRenderer / HtmlRenderer
```

**Rendering:** phase banners on `phase_change`; per-game board art (Risk world colored by owner + army counts; Diplomacy map with A/F glyphs + SC dots; SH liberal/fascist tracks + election tracker + government; ONW seat grid w/ revealed-or-hidden roles); a scrolling chat log tagged `[PUBLIC]` / `[P2→P5 whisper]`; an alliance ledger (`#7 {P2,P3} active` / `#7 ✓honored` / `#7 ✗BETRAYED by P3`); a footer betrayal/honour counter. Glyphs: 📢 public, 🔒 private, ⚑ propose, ✔ accept, ✘ decline, 💥 betray, ✅ honour (degrade to ASCII tags if no UTF-8).

**Hidden-info reveal:**
- `--god` (default): consume `setup.god_view` + `state_snapshot.hidden` for true roles; render every `observation` at full logged content (whispers marked "god-only"); show full alliance internals.
- `--seat N`: render seat N's `prompt` records verbatim + only `observation` records whose `audience` ∈ {contains N}; private messages N didn't receive appear only as `message_meta` (or nothing if `whisper_visibility="hidden"`); hidden roles hidden except N's own/peeks. **This is the visual QA that masking is correct.**
- `--follow`: open file, seek to end (or `--from-start --follow`), poll new lines, render incrementally, tolerate partial trailing line. Works against a live match because the runner flushes per `log()` call (open/append/close).
- `--html`: one self-contained file — inline CSS/JS, the JSONL embedded as `<script type="application/json">`, a scrub-bar over `event_id`, a client-side seat-selector that re-masks in the browser, and a final scoreboard + alliance graph.

---

## 7. File-by-file plan — 6 non-conflicting work units

**Conflict rule:** Unit S lands and is frozen first. No two units edit the same file. Units 1–5 import from S's modules but never edit them. Only S touches `base.py`, `runner.py`, `__init__.py`, `config.py`, `messaging.py`, `alliances.py`, `metrics.py`, `maps/*`, and the player files.

### Unit S — shared-infra + base + runner (merge first; publishes a fixture log + frozen interfaces)
- **Edit:** `play/base.py` (add `Obs`, `GOD`, default `Game.observations`/`god_view`/`snapshot`/`render_board`); `play/runner.py` (§2 routing courier, schema-v2 records, phase-change diff, steering stamping); `play/__init__.py` (export new symbols); `play/players/llm.py` + `play/players/steered_llm.py` (add `message`/`message_meta`/`alliance_event` cases to `_format_obs` / `receive_observation` — append only, no signature change).
- **New:** `play/config.py`, `play/messaging.py`, `play/alliances.py`, `play/metrics.py`, `play/maps/__init__.py`, `play/maps/risk_map.py`, `play/maps/diplomacy_map.py`, `play/maps/onw_roles.py`, `play/run.py` (thin CLI to launch a match so other units can produce logs).
- **New tests:** `tests/play/test_messaging.py` (routing/masking + INV-1 backward-compat with a stub Game), `tests/play/test_alliances.py` (lifecycle + summary math), `tests/play/test_runner_v2_schema.py` (every record has event_id/match_id; schema=2; v1 log replays), `tests/play/test_maps.py` (graph symmetry `b∈ADJ[a]⇔a∈ADJ[b]`, SC count==34, continent bonus totals).
- **Publishes:** a checked-in fixture log `tests/play/fixtures/sample_match.jsonl` for Unit 6.
- **Depends on:** nothing.

### Unit 1 — ONW parity + messaging + alliances
- **Edit:** `play/games/one_night_werewolf.py`. **New:** `tests/play/test_onw.py`.
- Compose `class OneNightWerewolf(MessagingMixin, AllianceMixin, Game)`; add `nego: NegotiationState`, `alli: AllianceState`, `roles` config to `ONWState`; implement §5.1; `living_seats`, `god_view`, `snapshot`, `render_board`; vote-bloc betrayal at `_resolve`; import `play/maps/onw_roles.py`.
- **Depends on (from S):** `Obs`, `GOD`, `MessagingMixin`, `NegotiationState`, `AllianceMixin`, `AllianceState`, `GameConfig`, `onw_roles.{ALL_ROLES,WAKE_ORDER,ROLE_COUNTS_BY_N,TEAM}`.

### Unit 2 — Secret Hitler parity + messaging + alliances
- **Edit:** `play/games/secret_hitler.py`. **New:** `tests/play/test_sh.py`.
- Compose mixins; add `nego`/`alli` to `SHState`; implement §5.2; `PH_DISCUSSION` before nomination; `god_view`/`snapshot`/`render_board`; coalition/gov-pact honour/betray at `_resolve_vote`/`enact`; `reveal` records for Investigate/Policy-Peek.
- **Depends on (from S):** same mixin/`Obs`/`GameConfig` surface as Unit 1.

### Unit 3 — Risk parity + messaging + alliances
- **Edit:** `play/games/risk_lite.py`. **New:** `tests/play/test_risk.py`.
- Compose mixins; full map via `play/maps/risk_map.py`; implement §5.3; `PH_NEGOTIATION`/`PH_FORTIFY`; `combat` records; `nonaggression`/`dmz` with `enforce_alliances`; `god_view`/`snapshot`/`render_board`.
- **Depends on (from S):** mixin surface + `risk_map.{TERRITORIES,ADJ,CONTINENTS,SET_VALUES}` + `GameConfig.enforce_alliances`.

### Unit 4 — Diplomacy parity + messaging + alliances (RISKIEST)
- **Edit:** `play/games/diplomacy_lite.py`. **New:** `tests/play/test_diplomacy.py` (DATC subset).
- Compose mixins; full map via `play/maps/diplomacy_map.py`; implement §5.4 with `adjudicate(orders, board) -> Resolution` pure function; Spring/Fall/Retreat/Winter phases; Full-Press `PH_NEGOTIATION`; support-pact/dmz honour/betray; `god_view`/`snapshot`/`render_board`.
- **Depends on (from S):** mixin surface + `diplomacy_map.{PROVINCES,KIND,SUPPLY_CENTRE,HOME_SC,ADJ_ARMY,ADJ_FLEET,COASTS,START_UNITS}`.

### Unit 5 — Watch/replay tool
- **New:** `play/viewer.py`, `scripts/watch_match.py`, `tests/play/test_viewer.py`.
- Pure log reader; static/follow/seat/god/step/html/metrics; builds a `name→Game` registry to call `render_board`. Tests assert god vs seat-N reveal differ correctly and no private text appears in a non-recipient seat render (golden-file over S's fixture log).
- **Depends on (from S):** **only the log schema (§4) + the fixture log.** Can start the moment S publishes the schema + fixture, in parallel with Units 1–4. (Board art for a given game requires that game unit merged, but the viewer falls back to the structured log if `render_board` returns '' — so it is not blocked.)

**Sequencing:** S → {1, 2, 3, 4, 5} fully parallel. S extends `tests/play/test_play_harness.py` (or adds it) with a `messaging=on, alliances=on` smoke arm per game using `RandomPlayer` (which picks legal `pass_talk`/random proposals).

---

## 8. Exact interface signatures Unit S must freeze (the parallel-build contract)

These are the only symbols Units 1–5 may depend on. Frozen at S-merge.

```python
# base.py
GOD: int = -1
Observation = dict
@dataclass
class Obs:
    audience: list[int]
    payload: dict
    log: dict | None = None
class Game(abc.ABC):
    config: "GameConfig"   # set in __init__ from optional arg
    def observations(self, prev_state, new_state, action: Action, actor: int) -> list[Obs]: ...
    def god_view(self, state) -> dict: ...
    def snapshot(self, state) -> dict: ...
    def render_board(self, state, *, reveal: str = "god") -> str: ...

# config.py
@dataclass
class GameConfig:
    messaging: bool = True
    alliances: bool = True
    nego_rounds: int = 2
    msgs_per_slot: int = 2
    max_msg_chars: int = 600
    enforce_alliances: bool = False
    whisper_visibility: str = "metadata"
    def as_dict(self) -> dict: ...

# messaging.py
MAX_MSG_CHARS_DEFAULT: int = 600
@dataclass
class NegotiationState:
    active: bool = False
    round_idx: int = 0
    max_rounds: int = 2
    speak_queue: list[int] = field(default_factory=list)
    budget: dict[int, int] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    return_phase: str = ""
class MessagingMixin:
    def start_negotiation(self, state, *, return_phase: str, rounds: int | None = None) -> None: ...
    def living_seats(self, state) -> list[int]: ...
    def nego_round_order(self, state) -> list[int]: ...
    def nego_active_player(self, state) -> int: ...
    def nego_legal_actions(self, state, player: int) -> list[Action]: ...
    def nego_parse(self, state, player: int, text: str) -> Action: ...
    def nego_step(self, state, action: Action): ...
    def nego_observations(self, prev_state, new_state, action: Action, actor: int) -> list[Obs]: ...
    def render_message_log(self, state, player: int) -> str: ...

# alliances.py
@dataclass
class Alliance: ...          # fields per §3.3
@dataclass
class AllianceState:
    next_id: int = 0
    alliances: dict[int, Alliance] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
class AllianceMixin:
    def alliance_legal_actions(self, state, player: int) -> list[Action]: ...
    def alliance_parse(self, state, player: int, text: str) -> Action: ...
    def apply_alliance_action(self, state, actor: int, action: Action) -> list[dict]: ...
    def alliance_observations(self, state, action: Action, actor: int) -> list[Obs]: ...
    def alliance_effects(self, state) -> dict: ...
    def judge_alliance(self, state, action: Action, actor: int) -> list[dict]: ...
    def is_allied(self, state, a: int, b: int) -> bool: ...
def alliance_summary(alli: AllianceState) -> dict: ...
def new_event(event: str, alliance: Alliance, *, turn: int, actor: int,
              counterparty: list[int], action_ref: dict | None = None) -> dict: ...

# metrics.py
def load_match(log_path: str) -> dict: ...
def steering_table(logs: list[str]) -> "Any": ...        # pandas DataFrame grouped by steering arm
def score_messages(log_path: str, judge_client) -> None: ...   # LLM judge (Sonnet+); emits msg_score; no regex

# maps/*  (plain data, per §8 tables in Designs B/C)
# risk_map:      TERRITORIES, ADJ, CONTINENTS, SET_VALUES, SET_INCREMENT
# diplomacy_map: PROVINCES, KIND, SUPPLY_CENTRE, HOME_SC, ADJ_ARMY, ADJ_FLEET, COASTS, START_UNITS
# onw_roles:     ALL_ROLES, WAKE_ORDER, ROLE_COUNTS_BY_N, TEAM
```

**Map data table shapes** (Unit S authors; Units 3/4/1 consume):
- `risk_map`: `TERRITORIES: list[str]` (idx=id), `ADJ: dict[int, frozenset[int]]`, `CONTINENTS: dict[str, tuple[frozenset[int], int]]`, `SET_VALUES: list[int]=[4,6,8,10,12,15]`, `SET_INCREMENT=5`.
- `diplomacy_map`: `PROVINCES: list[str]`, `KIND: dict[int,str]` (`"inland"|"coastal"|"sea"`), `SUPPLY_CENTRE: frozenset[int]` (len 34), `HOME_SC: dict[str,frozenset[int]]`, `ADJ_ARMY/ADJ_FLEET: dict[int,frozenset[int]]`, `COASTS: dict[int,list[str]]`, `START_UNITS: dict[str,list[tuple[str,int,str|None]]]`.
- `onw_roles`: `ALL_ROLES: list[str]`, `WAKE_ORDER: list[str]`, `ROLE_COUNTS_BY_N: dict[int,list[str]]`, `TEAM: dict[str,str]`.

---

## 9. Riskiest part — honest assessment (Diplomacy adjudication)

Full standard-Diplomacy order resolution on the 34-SC / 75-province map is the single hard, unbounded-risk deliverable. The current `diplomacy_lite._resolve_round` already has subtle bugs (the bounce-restore logic, lines 421–426, mishandles multi-unit swaps and circular movement). A naive resolver silently mishandles convoy paradoxes, beleaguered garrison, support-cut-by-dislodgement ordering, and self-dislodgement — and those silent rule violations would contaminate the steering metrics.

**Implement faithfully (tractable, needed for the betrayal signal):** 7 powers, 34 SCs, inland/coastal/sea typing + canonical adjacency (data, low risk), armies+fleets, Spring/Fall move + retreat + Winter build/disband, support + support-cut, standoffs/bounces, self-dislodgement ban, dislodged retreat-or-disband, builds/disbands from SC count, the full Full-Press negotiation/alliance layer (this is what the experiment measures, so it must be real).

**Simplify deliberately and document (in docstring + `match_start.config.adjudicator="lite-press-v2"`):** single-fleet convoys only (no chains; `ParseError` on chain orders); convoy paradoxes via the **Szykman rule** (treat as failed/hold — no fixpoint solver); split coasts modelled as single provinces; deterministic lowest-index retreat-or-disband; builds on home SC only with excess waived; documented deterministic tie/bounce for beleaguered-garrison and self-standoff nuances.

**Risk isolation:** put resolution behind a pure `adjudicate(orders, board) -> Resolution` and gate it with `tests/play/test_diplomacy.py` running the DATC cases that fall inside our subset (support cut, standoff bounce, single-fleet convoy, retreat-or-disband, build/disband counts) plus 6–10 alliance-betrayal scenarios; out-of-subset DATC cases are explicitly `pytest.mark.skip` with a reason. **Fallback if timeline slips:** keep the full map + phases + support + retreats/builds and stub convoys entirely — full-map Full-Press still yields the steering signal; flag the gap explicitly rather than ship a buggy adjudicator. Swapping in `pydip`/`diplomacy` behind the same `adjudicate()` is then a clean isolated follow-up that the messaging/alliance/observation layer does not depend on.

A secondary, smaller risk applies to all games: they mutate state in place, so the runner's `phase_change` diff and any future snapshotting MUST read `prev_phase` *before* `step` (handled in §2c). Unit S must not regress this.
