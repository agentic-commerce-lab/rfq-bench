# Concept: a Laya decision-model agent for rfq-bench

**Status:** implemented as the experimental `--agent laya` (see
[conditions](conditions.md#laya-decision-agent---agent-laya-experimental)); the cascade
and the alternative move designs are still design. **Date:** 2026-09-21.
**Author:** l.rump. **Depends on:** a Laya server (Laya by Convai Innovations,
Apache-2.0), which is not part of this repository.

## 1. Why

rfq-bench today compares three things that produce a **negotiation move**: deterministic
concession policies (`--agent scripted`), and a language model driven by a strategy or persona
(`--agent llm`, `--agent a2a`). All the LLM paths pay per-turn latency (seconds) and, on hosted
providers, money, and their moves are non-deterministic.

**Laya** is a different kind of decision-maker: a small (322–421M) **non-generative** encoder
that answers *typed questions* about a piece of text — `choice`, `score`, yes/no (`noul`) —
returning calibrated probabilities in ~20–35 ms on a laptop GPU, entirely locally, and
**deterministically** (one forward pass, no sampling). It cannot write a sentence, so it cannot
"generate" a counter-offer; it can only classify, score, and answer yes/no.

That mismatch is exactly what makes it an interesting fourth arm. The research questions:

1. **Can a non-generative decision model negotiate at all**, when the move is decomposed into
   typed decisions and the numeric work is done in code?
2. **How close does it get** to the LLM and scripted agents on captured value (`S_s`), agreement
   rate, and walk-away safety?
3. **What does it buy** — latency (~20–35 ms vs seconds), zero marginal cost, no data leaving the
   machine, and reproducible traces?
4. As a **cascade / guardrail** in front of the LLM negotiator, what escalation rate keeps
   value while cutting cost — the architecture the Laya skill recommends?

Success is **not** "Laya beats the LLM." Success is a clean, honest measurement of the
latency/cost/quality trade-off, and a working cascade. Laya's own upstream notes and an
independent eval put it strong on clear-cut classification (SMS spam 96%, news topic 93%) and
weak on ordinal/nuanced judgements (five-level rating 35%, six-way emotion 45%) — negotiation is
closer to the latter, so we expect it to need careful question design and cascading.

## 2. The core problem: a decision model that can't propose

A move is `(action ∈ {offer, accept, terminate}, package if offer)`. Laya can decide *action*
(a small classification) but cannot *construct a package* (that is generation). The skill is
explicit: **ask what the text says, not what to do; put state into words, never numbers; resolve
comparisons in code.** Negotiation is fundamentally numeric (prices, utilities, deadlines), so
most of the work is verbalizing state and doing arithmetic in code, leaving Laya a few
categorical decisions.

We keep rfq-bench's hard rules unchanged: the model never sees raw utilities/weights/BATNAs or
the opponent's private economics (the leakage guarantee), and any move is legal by construction
(snap-to-grid / reservation floor still apply as a backstop).

### Move design (built): Laya decides the offer

**Laya picks the offer itself (per-issue `choice`), and also owns accept/walk** — one forward
pass per turn:
- `accept` (`noul`): should the standing offer be accepted now? Gate `P(true)` on a threshold.
- `reachable` (`noul`): is a deal still reachable before the deadline? Drives `terminate`
  (feeds `walk✓` on no-ZOPA cases).
- one `choice` **per issue** over its legal values → Laya's picks form the offered package.

There is **no code-side concession schedule**: the offer is Laya's decision. The strategy is
injected into the state as **guidance text** (the same one-liner the LLM agent reads), so the
seven strategies form arms that can bias Laya's choice. Two faithful shop rails still apply
(legality, not strategy): a below-floor accept is refused, and a package Laya picks below the
reservation floor is raised to the floor.

*Design history.* An earlier build put concession in a code time-schedule (Laya only gating
accept/walk) because Laya returns a near-constant answer to "how much to concede" and has no
memory across turns. That was rejected as "code negotiating, not Laya." The current design lets
Laya decide the price, accepting the known cost: asking "which price do you propose" is a
"what to do" question that the skill warns invites **anchoring/inversion** — in practice Laya as
seller tends to anchor to the price already in the text rather than hold out. That is a
measurement to report, not a bug. Alternatives if this proves too weak: a package `choice` over a
few code-enumerated candidates (keeps coupling, one decision/turn), or reverting to the schedule
for a "Laya-gates-only" arm — both easy to add behind a switch.

## 3. Where it plugs into rfq-bench

Laya is just another thing that implements the negotiator protocol (`act(standing_offer, state)
-> Move`). Nothing downstream changes: scoring, ZOPA, traces, `report`, and the dashboard are
agent-agnostic.

```
agent/
  laya_client.py      # talk to the local Laya server (HTTP) or in-process laya.load()
  laya_negotiator.py  # LayaNegotiator: verbalize state -> typed questions -> Move
  laya_framing.py     # state -> Laya-friendly prose + code-computed comparisons + questions
  laya_settings.py    # base URL / checkpoint / thresholds (env-driven, like AgentSettings)
runners/offline.py    # already accepts an injected policy; no change needed
cli.py                # new --agent laya (and later --agent cascade)
```

- **Client.** The user is already running the Laya local server, so the default transport is an
  **HTTP sidecar** on loopback (`POST /predict {state, questions}` → JSON), mirroring how
  `LLMClient` targets a configurable base URL. `RFQ_BENCH_LAYA_URL` (e.g.
  `http://127.0.0.1:PORT`) points at it; an optional in-process mode (`laya.load(...)`) is a
  later convenience. One forward pass at a time on one GPU → the client serializes calls (a lock),
  so benchmark `--concurrency` will serialize on the Laya side (documented, not a bug).
- **Negotiator.** `LayaNegotiator` builds the verbalized payload + questions, calls the client,
  reads `answers`, applies the chosen move design, and returns a `Move` — same snap/floor rails
  and the same "unusable reply → hard error, episode excluded" contract as `LLMNegotiator`.
- **Traces.** Reuse `Trace` unchanged. `token_cost = usage.input_tokens`; `cost_usd = None`
  (local, free); `latency_s` measured per turn as today. Record the decision provenance
  per step in the existing `rationale`/`raw_response` fields: the chosen action, the winning
  option, and its probability/`confidence` (and `act_probability`) — so the dashboard replay and
  audit show *why* Laya moved, exactly as it shows the LLM's rationale.
- **Report / dashboard.** No change. Laya is a separate **agent condition**, so its run writes to
  its own file (`results/laya_v0.jsonl`) and is compared to `llm`/`scripted` side by side — not
  pooled (the benchmark forbids pooling across modes without a mode label). Cost overview shows
  tokens with "cost not reported" (correct — it is free).

## 4. Verbalizing the negotiation state

This is the make-or-break surface (the skill: "this matters more than anything else"). A new
`laya_framing.py` turns the numeric `NegotiationState` into prose, reusing `framing.py`'s
business-terms view and adding **code-computed conclusions** so Laya never has to compare numbers:

- Role, product, and the party's priorities in words (high/medium/low importance, preferred
  option order) — already produced by `framing.priorities_view`.
- The walk-away as a concrete limit ("the least favourable price you may accept is 110") —
  already produced by `framing.walk_away_view`.
- The standing offer described in words, plus a **precomputed verdict**: code compares the
  offer's own-utility to the reservation and to the current concession target and hands Laya the
  sentence ("this offer is slightly better than your bottom line but below what you're holding
  out for"). Numbers resolved in code, per the skill.
- Time as words, not a counter ("early in the negotiation" / "the deadline is close").
- No opponent private values, ever (leakage test extended to cover the Laya payload).

Questions come from `laya_negotiator` per the chosen design; we start from hand-written ones and
compare against Laya's ready-made sets where relevant, trying 2–3 phrasings and measuring (the
skill's method).

## 5. The cascade experiment (the compelling one)

Rather than only "Laya as a full negotiator," the architecture Laya is built for: **Laya as the
decision/guardrail layer, escalating hard turns.**

- `--agent cascade`: each turn, Laya first answers the gate questions. If a decision is clear
  (accept probability ≥ high threshold, or a confident stance), Laya's move stands. If it is
  **below threshold**, escalate that turn to the LLM negotiator (or to a deterministic policy),
  and record that the turn escalated.
- Report the **escalation rate** — the fraction of turns that paid LLM latency/cost — alongside
  `S_s`. The value proposition is captured value at a fraction of the LLM calls.
- Thresholds are chosen from labelled data (§6), not 0.5.

This turns "Laya vs LLM" into "how much of the negotiation can a 30 ms local model own before it
has to ask for help," which is the practically interesting number.

## 6. Evaluation & calibration plan

We reuse the whole scoring kernel and add operational metrics. Crucially, **rfq-bench already
knows the right answers**, which gives free labels Laya's skill says to collect:

- Ground-truth `accept`: the offer's own-utility ≥ the concession target (or ≥ BATNA) — from the
  private economics.
- Ground-truth `walk_away`: the no-ZOPA scenarios.
- These let us measure Laya's **decision accuracy** and **calibration (ECE)** on accept/walk, and
  **fit temperatures** on rfq-bench's own data before trusting probabilities (the skill's
  procedure), instead of guessing thresholds.

Metrics to report, per agent condition:
- The usual `S_s` / Δ (vs control), `agree%`, `joint`, `walk✓`, `rounds`.
- **Latency** per turn (expect Laya ~20–35 ms vs LLM seconds — the headline).
- **Decision accuracy & ECE** for accept/walk vs ground truth.
- **Escalation rate** (cascade).
- English checkpoint by default (calibrated); note the multilingual one is uncalibrated.

Report the measured numbers and the escalation rate, not "it works" (the skill's rule).

## 7. Determinism, reproducibility, validity

- **A plus over the LLM:** Laya is a deterministic forward pass, so — unlike nonzero-temperature
  LLMs — its traces are reproducible byte-for-byte given a fixed checkpoint and prompt. This fits
  the benchmark's determinism goal better than the LLM path.
- **Frozen dimensions.** The verbalization template and question set are frozen within a run and
  snapshotted (a Laya condition is a new versioned condition; the offline suite stays immutable).
- **Leakage.** The Laya payload goes through the same "no private economics, no opponent values"
  screen; extend `test_leakage.py` to the Laya framing.

## 8. Honest risks & limitations

- **Hybrid attribution.** In Designs A/C the package is chosen deterministically, so "the Laya
  agent" is Laya's gates + code's package. Attribute results precisely; do not claim Laya
  "negotiated the price" when code picked it. Design B is the only fully-Laya package path and is
  expected to be the weakest.
- **Ordinal/numeric weakness.** Laya is weak on graded judgements and cannot compare numbers;
  the design leans hard on verbalization and code-side arithmetic. If accept/stance accuracy is
  poor even after rephrasing and calibration, the honest outcome is "use it only as a
  high-threshold guardrail in a cascade," and we report that.
- **Verbalization bias.** Wording moves results a lot (the skill's inversion warnings). The state
  templates need the same 2–3-phrasing measurement Laya demands; treat them as part of the
  frozen condition.
- **Concurrency.** One forward pass at a time; benchmark parallelism serializes on the server.
- **Server dependence.** Requires the local server up (or in-process load, 25–35 s startup +
  warm-up). Runs should fail loudly if the server is unreachable.

## 9. Proposed build order (after this concept is approved)

1. `laya_client.py` (+ `laya_settings.py`): HTTP client to the local server, configurable URL,
   lock-guarded; a `laya doctor`-style probe that sends one `predict` and prints the answer.
2. `laya_framing.py`: state → prose + code-computed verdicts + the Design-A questions;
   extend the leakage test.
3. `laya_negotiator.py`: `LayaNegotiator.act` → `Move` (Design A), with snap/floor rails and the
   error contract; unit-test with a fake client.
4. `--agent laya` in the CLI + a `configs/laya-smoke.toml`; write `results/laya_v0.jsonl`.
5. Calibration/eval utilities: decision-accuracy & ECE vs ground truth; temperature fit.
6. `--agent cascade` + escalation-rate reporting.
7. Designs B/C behind a switch, measured against A.
8. Docs: the Laya section of `docs/conditions.md`; AGENTS.md scope note (new versioned condition).

## 10. Open questions for the user

- **Transport:** HTTP sidecar to your running server as the default (recommended), or in-process
  `laya.load(...)`? What's the server's URL/port and endpoint shape?
- **Checkpoint:** English (calibrated) is the default; do any scenarios need multilingual?
- **Scope of v1:** full `--agent laya` (Design A) first, or go straight for the `--agent cascade`
  guardrail since that's the strongest fit?
- **Move design priority:** start with A only, or build A and C together to A/B them early?
