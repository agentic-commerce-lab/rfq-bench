# Agent conditions

A **condition** is one fixed way of running the benchmark. Each writes to its own trace
file (and carries a `mode` on every trace), so results from different conditions are
never pooled. Pick one with `--agent` or `[run].agent`.

| `--agent` | Seller | Buyer | What varies | Default output |
|---|---|---|---|---|
| `scripted` | strategy policy | scripted opponent | strategy | `results/offline_v0.jsonl` |
| `llm` | LLM + strategy guidance (either role) | scripted opponent | strategy | `results/offline_v0.jsonl` |
| `a2a` | LLM + strategy guidance | LLM + persona | seller strategy, buyer persona | `results/a2a_v0.jsonl` |
| `duel` | model A or B | model B or A | the model | `results/duel_v0.jsonl` |
| `laya` (experimental) | Laya decision model | scripted opponent | strategy | `results/laya_v0.jsonl` |

## Scripted (`--agent scripted`)

The deterministic reference: the strategy policies from [Methodology](methodology.md#strategies)
play against the four scripted opponents. No model, no cost, byte-identical traces. Use
it to sanity-check scenarios and as the ground truth an LLM's strategy fidelity is
measured against.

## LLM agent (`--agent llm`)

One language model plays the tested side (buyer, seller or both via `--roles`), steered
by one line of strategy guidance, against the scripted opponents. The strategy is the
treatment; the model, prompt and decoding are fixed. See
[Getting started](getting-started.md#3-connecting-an-llm) for what the model sees and
how its moves are validated.

## A2A self-play (`--agent a2a`)

Both sides are language models (the same model):

- The **seller** is the scored agent, steered by a strategy (the treatment, with
  `control` as the baseline).
- The **buyer** is the opponent, steered by a **persona**: a behavioural disposition
  (tone, urgency, price sensitivity, risk tolerance, cooperativeness). The buyer's
  economics stay fixed by the scenario, so `q` is comparable across personas.
- **The sides can talk.** Each move may carry a short public `message` (up to 300
  characters delivered; the full text stays in the trace). A private `rationale` is
  recorded for you and never shown to the other side. Messages are cheap talk: only the
  offer binds, and a model may reveal, withhold or bluff.
- **Invalid moves are corrected and counted.** When snap-to-grid or the reservation
  floor changes a move, the corrected offer is played, that move's message is withheld
  (it quotes invalid terms), and the author is told in its own history what was played
  instead. The opponent only sees the corrected offer.

```bash
uv run rfq-bench run --agent a2a --strategies control,boulware,anchoring \
    --personas neutral,cost_focused,relationship_focused -y
uv run rfq-bench report --traces results/a2a_v0.jsonl
```

Both sides are scored from the same traces: `report` prints the seller table (by
strategy, Δ vs `control`) and a buyer table (by persona, Δ vs `neutral`). The dashboard
has a seller/buyer toggle. Every round makes two API calls, so cost roughly doubles.

### Personas

**Procurement-grounded personas** (defaults in `configs/a2a-grid.toml`, defined in
`prompts/personas/*.md`), each mapping a recognized negotiation motive:

| Persona | Archetype | Basis | Sources |
|---|---|---|---|
| `cost_focused` | Leverage / cost-focused buyer | Distributive motive: high aspiration, strong BATNA, little information sharing, price extraction. | Kelly & Chicksand (2024); Sebenius (2017) |
| `total_value` | Integrative / total-value buyer | Problem-solving orientation: several issues, shares preferences, looks for trade-offs. | Kelly & Chicksand (2024); Elgoibar et al. (2021) |
| `reliability_first` | Risk- and compliance-first buyer | Loss sensitivity: pays more for delivery certainty, quality or compliance. | Kahneman & Tversky (1979); Choudhary et al. (2023) |
| `relationship_focused` | Relationship / continuity buyer | Values reliability, reciprocity and future continuity beyond this price. | Kumar et al. (2025); Elgoibar et al. (2021) |

**Price-only set** (`configs/a2a-price.toml`). With price as the only issue, personas
defined by trade-offs against other issues have nothing to trade (`total_value` and
`reliability_first` behaved like `neutral`). The price suite uses `neutral`,
`cost_focused`, `relationship_focused`, plus:

| Persona | Archetype | Basis | Sources |
|---|---|---|---|
| `supply_security` | Risk-first buyer, single-issue form | The risk is not closing the deal; loss aversion shows as paying a premium for certainty, within the walk-away. | Kahneman & Tversky (1979); Choudhary et al. (2023) |
| `time_pressured` | Time-pressured buyer | Delay is costly: prefers a quick acceptable deal and pays more near the deadline. | Carnevale & Lawler (1986) |

There are also generic built-in dispositions for quick checks (`neutral`,
`bargain_hunter`, `time_pressured`, `relationship_builder`, `hardball`, `risk_averse`).

References: Carnevale & Lawler (1986), *Time pressure and the development of integrative
agreements in bilateral negotiations*, J. Conflict Resolution 30(4). Choudhary et al.
(2023), *Risk assessment in supply chains*. Elgoibar, Munduate & Euwema (2021),
*Increasing integrative negotiation through trustworthiness and trust*. Kahneman &
Tversky (1979), *Prospect theory*, Econometrica 47(2). Kelly & Chicksand (2024), *A
critical exploration of bargaining in purchasing and supply management*. Kumar et al.
(2025), *Managing buyer experience in a buyer–supplier relationship*. Sebenius (2017),
*BATNAs in negotiation: Common errors and three kinds of "no"*.

## Model duel (`--agent duel`)

A duel answers "which of two models negotiates better?". Self-play cannot: swapping the
model changes the buyer too, and on zero-sum price scenarios a stronger buyer cancels a
stronger seller. In a duel the two models negotiate **against each other, in both roles**.

```bash
uv run rfq-bench run --config configs/duel-sol-opus.toml
# or
uv run rfq-bench run --agent duel --models openai/gpt-6-sol,anthropic/claude-opus-5.5 \
    --data data/scenarios_price --pairings full
```

- **Pairings.** `cross`: A sells to B and B sells to A. `full`: adds A vs A and B vs B
  (self-play), at twice the cost.
- **The model is the treatment.** Every cell (scenario × strategy × persona × first
  speaker) is played once per pairing. Strategy and persona default to `control` and
  `neutral`; decoding, prompts and `tool_choice` are shared.
- **Model score.** Per cell, a model's score is the mean of its seller `q` and its buyer
  `q` against the other model:

  ```
  S_A = ( q_seller[A sells to B] + q_buyer[B sells to A] ) / 2
  ```

  Each pairing has one seller and one buyer, so a role advantage built into a scenario
  cancels, and each model always faces the other. `Δ = S_B − S_A`, with a
  scenario-clustered 95% CI.
- **Skill by role** (`full` only). Selling skill compares A and B selling to the same
  buyer; buying skill compares them buying from the same seller.

`rfq-bench report` prints the duel readout (overall, by role, by scenario, per matchup,
cost by model, serving providers). The run writes a [duel dashboard](reports-and-dashboards.md#duel-dashboard).

A duel ranks two models on one setup. It is not a general leaderboard, and it inherits
A2A's limits (prompt-level strategies and personas, nondeterminism).

## Laya decision agent (`--agent laya`, experimental)

**Status: experimental. Requires a Laya server, which is not part of this repository.**

Laya (by Convai Innovations, Apache-2.0) is a small,
local, non-generative decision model that answers typed questions (`choice`, yes/no) in
one forward pass: deterministic, about 20–35 ms per turn, no API cost. As the tested
seller, Laya picks each issue's value itself (one `choice` per issue) and decides
accept or walk. There is no code concession schedule; the strategy is guidance text Laya
reads. Two shop rails keep moves legal (a below-floor accept is refused, a below-floor
pick is raised to the floor).

Point `RFQ_BENCH_LAYA_URL` at a compatible server (`POST /predict`; default
`http://127.0.0.1:8000`). Accept and walk thresholds are `RFQ_BENCH_LAYA_ACCEPT_THRESHOLD`
and `RFQ_BENCH_LAYA_WALK_THRESHOLD`.

```bash
uv run rfq-bench run --config configs/laya-smoke.toml
uv run rfq-bench report --traces results/laya_v0.jsonl --control laya
```

"Which price do you propose" invites anchoring (as seller Laya tends to match the price on
the table); that is a measurement, not a bug. Design, limitations and the planned cascade
experiment: [`laya-agent-concept.md`](laya-agent-concept.md).
