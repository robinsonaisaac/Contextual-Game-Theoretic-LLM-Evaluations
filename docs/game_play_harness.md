# Multi-game LLM play harness — design

A plug-in framework for running LLMs (including activation-steered LLMs) as
players in real strategic games. Goal: take a steering vector fit on the
vignette corpora (cooperation, trust, …) and measure whether it transfers
to held-out games that the corpus did not see.

## Goals

1. **One game-runner, many games.** Same code path runs Werewolf, Coup,
   Diplomacy, Risk, Catan, Poker, and Monopoly. New games plug in by
   implementing a small `Game` interface.
2. **One player interface, many players.** Drop-in support for
   API-routed LLMs (OpenRouter), local LLMs (Modal-hosted Gemma 4 via
   `SteeringWorker`), and heuristic baselines.
3. **Steering hooks.** A `SteeredLLMPlayer` is a thin wrapper around
   `LLMPlayer` that activates an existing steering vector for the duration
   of one game; comparing steered vs unsteered match-ups gives us a
   transfer measurement.
4. **Reproducible matches.** Every game is logged as a JSONL trace
   (one row per action, plus metadata) so matches can be replayed and
   aggregated. Seeds for both game state and LLM sampling are recorded.
5. **Cheap before expensive.** Add Poker first (one decision per street,
   mature eval literature). Build up to Werewolf/Diplomacy.

## Non-goals (for v1)

- Real-time game UIs; this is batch evaluation.
- Self-play training; only inference-time matches.
- Beating SOTA bots in any game. We care about *steering* effects, so a
  consistent baseline is enough.

## Architecture

```
game_theory_llm/play/
    base.py             abstract Game / Player / Match / Action
    runner.py           the match-loop + JSONL logger
    actions.py          dataclasses for parsed action structures
    games/
        __init__.py
        poker_heads_up.py
        coup.py
        werewolf.py
        diplomacy.py
        risk.py
        catan.py        (deferred — needs trade engine)
        monopoly.py     (deferred — slot for completeness)
    players/
        __init__.py
        llm.py          OpenRouter / OpenAI-style API player
        steered_llm.py  Modal-backed SteeringWorker + steering hook
        random.py       uniform-random baseline
        rule.py         simple heuristic baselines per game
```

### `Game` interface

```python
class Game(Protocol):
    name: str
    n_players: int                                # exact or min/max
    def initial_state(self, rng: Random) -> State: ...
    def legal_actions(self, state: State, player: int) -> list[Action]: ...
    def render_prompt(self, state: State, player: int) -> str: ...
        # Builds a textual game-state description for that player only
        # (hides hidden info). Returns the full prompt to send to the LLM.
    def parse_action(self, state: State, player: int, text: str) -> Action: ...
        # Parses the LLM's text response into a legal Action. May raise
        # ParseError; runner has a retry budget per action.
    def step(self, state: State, action: Action) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def rewards(self, state: State) -> list[float]: ...
        # final scalar reward per player at terminal states.
```

The `State` itself is a per-game dataclass; the runner treats it opaquely.

### `Player` interface

```python
class Player(Protocol):
    name: str
    def act(self, game: Game, state: State, player_idx: int) -> Action: ...
    def receive_observation(self, obs: dict) -> None:
        # default no-op; chat-based players append to their conversation
```

### Runner

```python
def run_match(game: Game, players: list[Player], *,
              seed: int, log_path: Path,
              max_turns: int = 10_000) -> MatchResult: ...
```

The runner owns the JSONL log:
```
{"turn": 0, "type": "state", "state": {...}, "active": 0}
{"turn": 0, "type": "prompt", "player": 0, "text": "..."}
{"turn": 0, "type": "action", "player": 0, "raw": "...", "parsed": {...}}
{"turn": 0, "type": "step", "state_after": {...}}
...
{"type": "terminal", "rewards": [1.0, 0.0]}
```

This format lets us replay matches, slice statistics by turn/player/action
type, and compute aggregate steering-effect metrics offline.

### Steered LLM player

```python
class SteeredLLMPlayer(LLMPlayer):
    """Thin wrapper that holds the steering parameters and dispatches each
    LLM call to the Modal SteeringWorker with the hook active."""
    model_short: str           # e.g. "E4B"
    vectors_run_id: str        # e.g. "pd_E4B_v1" or "trust_E4B_v1"
    layer: int                 # e.g. 16
    position: str              # "mean_trace" | "last_prompt" | "last_trace"
    alpha: float               # steering coefficient
```

Each `act()` call constructs the prompt as the unsteered player would,
then dispatches to `SteeringWorker.eval_shard.spawn(...)` with a single-
story list, recovers the parsed decision, and steps the game. We
amortise model-load by keeping a long-lived `SteeringWorker` instance per
(model, layer, position, alpha) combination per match.

For OpenRouter/API players we cannot apply steering — those serve as
baseline / opponent slots.

## Per-game notes

Action spaces, hidden-info structure, and likely difficulty in priority
order:

### Heads-up No-Limit Hold'em (`poker_heads_up.py`)

- Smallest action space: {fold, check, call, raise(amount)}.
- One decision per street; ~4 decisions per hand.
- Hidden info = opponent's hole cards.
- **Eval design:** N hands per matchup, measure bb/100 won by the steered
  player vs the unsteered player. Public infrastructure: PokerKit (the
  pokerlib follow-up) handles state and legal-action enumeration.
- **Why first:** action parsing is trivial; matches are short; steered
  players can play a thousand hands cheaply.

### Coup (`coup.py`)

- 5-card deck (Duke, Assassin, Captain, Ambassador, Contessa).
- Action set per turn: ~10 (income, foreign aid, coup, tax, assassinate,
  steal, exchange, plus challenge/block reactions).
- Hidden info = your two roles. Bluffing is core: you can *claim* any
  role regardless of what you hold.
- **Eval design:** ~50 4-player matches per condition; track win rate of
  steered seat. Pair steered seat vs three unsteered LLM seats.
- **Trust angle:** a paranoid (low-trust steered) model challenges
  claimed roles more often; a trusting model under-challenges and gets
  bluffed.

### Werewolf / Mafia (`werewolf.py`)

- Roles: Villager, Werewolf, Seer, Doctor, sometimes Hunter / Tanner.
- Multi-turn: night phase (private) + day phase (public discussion +
  vote).
- Hidden info = roles. Day phase is free-form text chat.
- **Eval design:** ~30 games per condition. Track survival rate of
  steered Villagers, and how often steered Werewolves successfully avoid
  detection. Use the published Werewolf-Bench framework if available;
  otherwise implement the 7-player canonical configuration.
- **Operational gotcha:** the day-phase chat is multi-turn within a
  single round, so each Villager makes multiple LLM calls before voting.
  Per-game LLM-call budget is the main cost driver.

### Diplomacy (`diplomacy.py`)

- 7-player simultaneous-move game on a Europe map.
- Two phases per turn: messaging (free-form negotiation), then orders.
- Public infrastructure: `diplomacy` (the FAIR-released engine), DipNet,
  No-Press Diplomacy, etc.
- **Eval design:** No-Press variant first (skip messaging) so we measure
  pure trust without language complexity; then Full-Press where steering
  may affect both messaging tone and order-issuing.
- **Operational gotcha:** matches are long (~10 turns × 7 players × 2
  phases). Plan for ~$100 per matchup at frontier-model prices.

### Risk (`risk.py`)

- Territorial. Alliances are optional and informal — trust matters in
  the negotiation phase but not the dice phase.
- Public infrastructure: a few open-source engines (`risk-python`,
  `risk-go`); we'd wrap one rather than reimplement.
- **Eval design:** 6-player matches, 30 per condition. Track survival
  duration of steered seat and number of successful alliances.
- **Lower priority:** trust signal is dilute; many turns are pure tactics.

### Catan and Monopoly (deferred)

- Catan: trading is the trust-relevant mechanic; needs a robust offer-
  acceptance subsystem.
- Monopoly: low trust signal (mostly luck + accounting). Including for
  completeness, not as a primary transfer test.

## Suggested implementation order

1. **`base.py` + `runner.py` + `LLMPlayer` + `RandomPlayer`** (1–2 days)
2. **Poker heads-up + PokerKit integration + a rule-based "tight-passive"
   baseline opponent** (1 day)
3. **`SteeredLLMPlayer` wrapping the existing `SteeringWorker`** (0.5 day)
4. **First transfer experiment**: trust-steered Gemma 4 26B-A4B vs
   unsteered self in 1000-hand heads-up Hold'em (compute: ~$20)
5. **Coup** (2–3 days; the bluff/challenge logic is the bulk of it)
6. **Werewolf** (3–4 days; or wrap an existing framework if available)
7. **Diplomacy** (1 week; substantial state-rendering work)
8. Risk, Catan, Monopoly as later additions

## Metrics

For each (game, steering condition, opponent) tuple we compute:

- **Outcome metric** (per game): win rate, bb/100, survival rate.
- **Behavioural metric** (per game): the action-distribution shift caused
  by steering, e.g. challenge rate in Coup, fold rate in Poker, betrayal
  count in Diplomacy.
- **Confidence intervals** via bootstrap over matches.

The interesting questions across games are:
- Does *trust-steered* mean more/less trusting on each game?
- Do the effect sizes scale with model size, as they did on the vignette
  corpus?
- Does a single steering direction generalise across all games, or do we
  need game-specific vectors?

The harness is what lets us answer those at all.

## Open questions

- **Match-level steering caching.** Loading a `SteeringWorker` per
  `(model, layer, alpha)` is fine if we batch many matches; in a 1000-hand
  Poker session we keep the container warm. For game-agnostic experiments
  we may want to amortise across games too.
- **Hidden-state leakage.** Some games (Werewolf, Coup) need careful
  prompt rendering to avoid revealing other players' hidden info. The
  `render_prompt(state, player)` interface enforces this.
- **Multi-turn steering integrity.** Currently `SteeringWorker.eval_shard`
  applies the hook for one generation. Within a multi-turn match each
  `act()` call is a fresh generation; the hook will apply consistently.
  We should verify the model's chat history fed back in still produces
  steered behaviour on later turns.
- **Whether to steer opponents.** First study: steer only one seat.
  Follow-up: steer multiple seats with the same/opposite vectors to
  isolate the interaction effects.
