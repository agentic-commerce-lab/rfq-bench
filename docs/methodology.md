# Methodology

rfq-bench is built around one controlled comparison: a fixed agent negotiates, and only
the thing under test changes. In the core benchmark that thing is the **strategy**; in a
[model duel](conditions.md#model-duel---agent-duel) it is the **model**. Everything else
(prompt, tool schema, decoding settings, the information the agent sees, the outcome
space) is frozen and recorded on every trace.

## The score

For the scored party *i* in episode *e*, with realized utility `U`, walk-away (BATNA)
utility `d` and ideal attainable utility `I`:

```
q = clip( (U − d) / (I − d), 0, 1 )
```

`q` is the share of the available gain the party secured: `0` means no better than
walking away, `1` means its best possible deal. A no-deal falls back to the BATNA, so
`q = 0`.

- **Degenerate scenarios** (`I == d`) have no defined `q` and are excluded, never scored 0.
- **No-ZOPA scenarios** (no outcome beats both walk-aways) are excluded from the score.
  There a correct walk-away and a below-BATNA accept both give `q = 0`, so they carry no
  signal about strategy or skill. They are reported as a safety result (`walk✓`).
- **Errored episodes** (the agent produced no usable move) are excluded and counted.

The **strategy score** macro-averages `q` over every matched cell (scenario × opponent ×
role × first speaker × repetition), each cell weighted equally:

```
S_s = 100 × mean over cells of q
```

The **effect** of a strategy is its difference to the identical control condition, with
a 95% confidence interval from a bootstrap that resamples **scenarios** (clusters):

```
Δ_s = S_s − S_control
```

For an arm that only runs on some scenarios (e.g. `logrolling`, multi-issue only),
control is restricted to the same scenarios. Scoring runs on the persisted traces, never
inside the negotiation engine, so every number can be recomputed and audited without
calling a model again.

The score measures value captured above the walk-away and nothing else. Fairness, joint
surplus, efficiency, safety, speed and cost are reported as separate tracks and never
folded in.

## How an episode runs

Episodes run on a [NegMAS](https://negmas.readthedocs.io) `SAOMechanism` (stacked
alternating offers).

1. **Build the game.** The scenario's issues become NegMAS issues; the outcome space is
   their Cartesian product. Each party gets an exact weighted-additive utility function
   with its reservation value set to its BATNA.
2. **Wire the negotiators.** The tested agent (a strategy policy, an LLM negotiator or
   Laya) and the opponent are wrapped as NegMAS negotiators and added in first-speaker
   order.
3. **Run.** Offers and counter-offers alternate until an acceptance, a walk-away or the
   deadline. Each side sees only its own preferences and the public offer history.
4. **Record.** Offers, messages, the agreement (or no-deal), realized utilities,
   validity flags, latency, tokens, cost and provenance go into an immutable JSONL trace.

A full offline run is a balanced matrix: every cell is mirrored across which role the
tested agent plays and who opens, and swept over strategies × opponents × scenarios ×
seeds. Scripted policies are deterministic, so their traces are byte-identical across
reruns.

## Strategies

Each strategy is a deterministic concession policy from the time-dependent family of
Faratin, Sierra & Jennings (1998). It slides a target utility from an opening toward the
BATNA over the deadline:

```
target(round) = BATNA + opening · (1 − (round / (deadline − 1))^(1/β)) · (ideal − BATNA)
```

Each round the agent offers the most generous outcome that still meets its target, and
accepts a standing offer once it clears the target (never below the BATNA). This is the
standard concession-tactic family used in ANAC and in the GENIUS and NegMAS platforms.

| Strategy | opening | β | Behaviour | Source |
|---|:-:|:-:|---|:-:|
| `control` | 0.85 | 1.0 | Neutral baseline: steady concession, no anchoring | — |
| `boulware` | 1.00 | 0.05 | Holds near the opening, concedes sharply at the deadline | [1] |
| `linear` | 0.90 | 1.0 | Concedes at a roughly constant rate | [1] |
| `conceder` | 0.90 | 5.0 | Concedes rapidly and early | [1] |
| `tit_for_tat` | 0.85 | — | Opens cooperatively, then mirrors the opponent's concession | [2] |
| `anchoring` | 1.00 | 1.0 | Ambitious but feasible opening, then a fixed linear slope | [3] |
| `logrolling` | 0.90 | 1.0 | Trades across issues toward the Pareto frontier (multi-issue only) | [4] |

`tit_for_tat` is not time-based: its target reciprocates the opponent's measured
concession on the agent's own utility scale. `logrolling` uses the linear schedule but,
among offers that meet its target, prefers the values the opponent last asked for, so it
concedes on issues it values least without seeing the opponent's utilities.

The scripted **opponents** use the same machinery with fixed parameters: `hardliner`
(opening 1.0, β 0.02), `fast_conceder` (0.9, 6.0), `reciprocal` (tit-for-tat) and
`integrative` (0.9, 1.0, logrolling).

With an LLM or Laya agent, the strategy becomes one line of guidance text the agent
reads. Whether the agent actually plays it is checked by the fidelity analysis below.

## Scenario sets

| Dataset | Scenarios | Contents |
|---|---|---|
| `data/scenarios/` | 5 | The v0.1 anchor set: 2 single-issue price cases, 2 multi-issue cases (price, delivery, warranty), 1 no-ZOPA diagnostic. Immutable. |
| `data/scenarios_price/` | 8 | Price-only suite: 21 price tiers each, linear utilities, walk-aways on a tier. Six with a ZOPA (wide, narrow, skewed low and high, uneven tiers, high-value contract) and two without. |

Scenarios define products, issue values, each party's weights and utilities, BATNAs and
deadlines. Private economics never enter a model-visible payload.

## Metrics

**Score track** (the comparison): `q`, `S_s` and `Δ` as above.

**Separate tracks** (reported alongside, never folded in):

| Metric | Meaning |
|---|---|
| `agree%` | Share of episodes that reached a deal. Includes no-ZOPA cases, where no deal is correct. |
| `joint` | Mean joint surplus of agreements, `(U_buyer − d_buyer) + (U_seller − d_seller)`. Near-constant on single-issue (zero-sum) scenarios. |
| `pareto%` | Share of agreements on the Pareto frontier. |
| `rounds` | Mean actions to close. NegMAS logs about two actions per round. |
| `walk✓` | On no-ZOPA scenarios, the share of episodes that correctly ended without a deal. |
| latency, tokens, cost | Per episode. Real USD cost when the provider reports it (OpenRouter's `usage.cost`), otherwise "not reported". |
| policy constraints | How often snap-to-grid or the reservation floor changed a move, by role and arm. |

Per-episode validity flags are stored on each trace: `agreement_reached`,
`legal_agreement`, `no_batna_violation`, `correct_walk_away`.

### Strategy fidelity

An LLM or Laya agent only reads its strategy as text, so it may not play it. `report`
replays each arm's definition (opening, concession shape, acceptance rule) over the
agent's actual history and compares turn by turn:

- **`open` / `bias` / `|gap|`**: agent offer minus the strategy's offer, as a share of the
  agent's BATNA-to-ideal range. Positive means tougher than the strategy.
- **`on-tgt`**: share of offers within ±0.10 of the strategy's offer.
- **`early✗` / `missed✗`**: accepts the strategy would not make, and accepts it would have
  made.
- **`β fit` / `β ref`**: the concession shape fitted to the agent's offers vs the
  strategy's own.

Scripted agents replay with zero gap by construction. If arms show a Δ but near-identical
fidelity profiles, the model is not really differentiating the strategies.

## Reading the results

| Metric | Scale | How to read it |
|---|---|---|
| `S_s` | 0–100 | Share of the available gain captured. About 50 is an even split. Its meaning comes from the gap to control. |
| `Δ` | points | `S_s − S_control`. Positive beats control. |
| 95% CI | points | Excludes 0: a real effect. Includes 0: indistinguishable from control. This decides whether Δ means anything. |
| `agree%`, `walk✓`, `pareto%` | 0–100% | Absolute; judge them on their own. |
| `joint`, `rounds`, cost | relative | Compare arms with each other, not with a fixed threshold. |

- Read **Δ with its CI**, not `S_s` alone. `S_s = 38` against a control of 35 with a CI of
  `[+1, +5]` is a small, real gain; `S_s = 45` against 48 is a loss.
- **Check the exclusions first.** A run with many errors or degenerate cells has a small
  `n` and shaky numbers.

Worked example from a scripted run: `boulware S_s = 8, Δ = −14.8, CI = [−19.7, −7.3],
agree% = 20%`. Boulware captured far less than control, the loss is significant, and the
reason is visible in the agreement rate: it walked away most of the time, holding out for
deals it rarely got.

## References

1. Faratin, Sierra & Jennings (1998). *Negotiation decision functions for autonomous
   agents.* Robotics and Autonomous Systems 24, 159–182.
2. Baarslag, Hindriks & Jonker (2013). *A Tit for Tat Negotiation Strategy for Real-Time
   Bilateral Negotiations.*
3. Galinsky & Mussweiler (2001). *First offers as anchors.* JPSP 81, 657–669.
4. Tajima & Fraser (2001). *Logrolling Procedure for Multi-Issue Negotiation.* Group
   Decision and Negotiation 10, 217–235.
5. Mohammad, Nakadai & Greenwald (2021). *NegMAS: A Platform for Automated Negotiations.*
   https://doi.org/10.1007/978-3-030-69322-0_23
