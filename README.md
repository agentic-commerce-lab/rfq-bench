# rfq-bench

A reproducible **within-agent negotiation benchmark** for buyer/seller language
agents in e-commerce. One fixed agent negotiates against deterministic opponents
in controlled bilateral SAO sessions; **only the strategy policy varies**. The
score measures how much value each strategy captures above the agent's BATNA.

See [`AGENTS.md`](./AGENTS.md) for the full build guide and the scientific
contract. This is POC v0.1 (offline benchmark + the iteration-1 smoke milestone).

## Quickstart

```bash
uv sync --extra dev                 # create the environment
uv run rfq-bench list               # strategies, opponents, scenarios
uv run rfq-bench run --overwrite    # shows the run plan + episode count, then asks to confirm
                                    # (add -y / --yes to skip the prompt in scripts)
uv run rfq-bench report             # score the traces and print the effects table
uv run rfq-bench dashboard          # write results/dashboard.html (interactive, offline)
uv run pytest                       # run the test suite (no network required)
```

## Reusable config

Instead of retyping the matrix and run flags every time, capture them in a TOML
file:

```bash
uv run rfq-bench config init          # scaffold ./rfq-bench.toml (commented)
# edit it, then:
uv run rfq-bench run --config rfq-bench.toml --overwrite
uv run rfq-bench config show           # validate a config and print its settings
```

An `rfq-bench.toml` in the current directory is **auto-discovered** (no `--config`
needed). Precedence is **explicit CLI flags > config file > built-in defaults**, so
the file sets your reusable baseline and any flag you type still overrides it. Unknown
keys are rejected so typos fail loudly. Example:

```toml
[run]
scenarios   = ["single_price_a", "multi_pdw_a"]
strategies  = ["control", "boulware", "anchoring"]
opponents   = ["hardliner", "reciprocal"]
seeds       = [0, 1]
agent       = "llm"
concurrency = 8            # stagger auto-ramps parallel starts (--stagger to tune)
out         = "results/offline_v0.jsonl"

[dashboard]
control = "control"        # scoring knobs for the post-run dashboard refresh
n_boot  = 2000
```

## Interactive dashboard

`rfq-bench dashboard` scores the traces with the same kernel as `report` and
writes one self-contained HTML file — open it with a double-click, no server.
It has a filterable leaderboard with a Δ-vs-control forest plot, a **cost overview**
(total/mean spend and tokens, broken down by model and strategy — real USD when the
provider reports it), a strategy×cell heatmap of mean `q`, an episode browser, and a
round-by-round negotiation replay
(offer/counter-offer vs the ZOPA band, with a step scrubber). When the LLM agent's
move is **adjusted** (off-grid value snapped to a legal tier, or a below-floor quote
clamped to the shop floor) or **errored** (episode excluded), the replay flags the
step, names **why**, and shows **what the model tried** vs what was played — no
scripted values are ever substituted. Each LLM turn also exposes the model's
**full verbatim reply** and its **chain-of-thought** (when a reasoning model returns
one) as collapsible panels, so you can verify exactly what happened. Filters recompute
scores and scenario-clustered bootstrap CIs live in the browser; per-episode `q`
comes from the Python kernel, so the numbers always match `report`.

## Using an LLM agent

The fixed agent talks to any **OpenAI-compatible** endpoint and submits every move
via **real function/tool calling** (a forced `submit_move` call whose arguments are
constrained to the legal options) — so the tested model **must support function
calling**. Copy `.env.example` to `.env` and set `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
and `RFQ_BENCH_MODEL`, then:

```bash
uv run rfq-bench run --agent llm --overwrite
```

The LLM's decisions stand — **no scripted-strategy value is ever substituted for
the model**. The model sees the scenario in **business terms** (its role, the
product, each issue's options and units, its priorities, and its walk-away /
bottom line) — never the raw 0–1 utilities. Its move is adjusted only by two
real-shop rules, and otherwise recorded as an error:

- **Snap to grid** — an off-grid value is projected onto the nearest legal tier
  (like an ordering system rounding to an orderable option), logged as an
  adjustment.
- **Shop reservation floor** — a quote below the agent's own reservation is
  blocked by policy and clamped up to the floor (a static business limit, not a
  strategy). Since the agent is *told* its floor, this is a rare backstop.
- **Hard error** — an unusable reply (unparseable, missing an issue, a value that
  can't be snapped) is logged and the episode is **excluded from scoring**; the
  error rate is reported separately.

Bad-but-legal decisions above the floor are played and scored as the model's own.
Decoding config is frozen and snapshotted into every trace.

Runs are resilient to flaky endpoints: an episode that errors (a transient API or
network failure) is re-attempted with a short backoff (`--retries`, default 2), and
the run reports how many episodes recovered vs. still failed. Malformed JSON from the
model no longer fails an episode — it falls back to the deterministic policy, with the
raw reply preserved in the trace for inspection in the dashboard.

### Custom system prompts

The agents' **system prompt** is a markdown file. The default is shipped with the
package (`rfq_bench/agent/prompt_templates/system.md`); override it without touching
code, per side (buyer / seller) if you like. Resolution order, highest first:

1. `RFQ_BENCH_BUYER_SYSTEM_PROMPT` / `RFQ_BENCH_SELLER_SYSTEM_PROMPT` (a role-specific
   file path in `.env`),
2. `RFQ_BENCH_SYSTEM_PROMPT` (a shared file path),
3. `prompts/system_<role>.md` in the working dir (auto-discovered),
4. `prompts/system.md` (auto-discovered, shared),
5. the packaged default.

The quickest customization: copy `prompts/system.example.md` to `prompts/system.md`
(or `prompts/system_buyer.md` / `prompts/system_seller.md`) and edit — no env needed.
By default both roles share the packaged prompt. In A2A a role-specific prompt is an
asymmetry between the sides, so keep any per-role files mirrored.
See [`prompts/README.md`](./prompts/README.md).

**The per-turn payload** is compact JSON in cache-friendly order. Content that's fixed
for the episode comes first (role, each issue's options in preference order, the
walk-away), then the arm's `approach_instruction`, then an append-only `history` of
`[who, offer, message?]` entries, and last the fields that change every turn
(`rounds_left`, the standing offer). Providers that cache prompt prefixes reuse
everything up to the newest move. `rfq-bench report` prints the measured cache hit
rate, and an A2A messages summary, whenever the traces carry them.

**Strategies and personas are markdown too.** Drop `prompts/strategies/<name>.md` or
`prompts/personas/<name>.md` (the filename stem is the name) to add a new strategy /
persona or override a built-in — the file's text is the one-line guidance the LLM
receives. `rfq-bench list` shows the effective set (new names marked `*`); name one in
a run with `--strategies …,<name>` / `--personas …,<name>`. A markdown strategy defines
only the LLM guidance (works with `--agent llm`/`a2a`); a brand-new strategy in scripted
offline mode also needs a policy in `strategies/registry.py`. Keep custom personas
disposition-only (no private economics) — the leakage test guards the built-ins, not
your files. Details in [`prompts/README.md`](./prompts/README.md).

The prompt is frozen *within* a run
(identical text for every arm); a custom prompt is a new, explicitly versioned
condition — don't compare its results against runs made with a different prompt. Use
`rfq-bench doctor --role seller` to see which prompt resolved. **Caveat:** the resolved
prompt text is not yet snapshotted into traces (a known provenance gap — see
Limitations), so record which prompt you used alongside the results.

## Strategy arms

A **strategy** is the only thing that varies between matched runs — the model,
prompt, tools, decoding, protocol, and outcome space are all frozen. Each
strategy is a deterministic **concession policy**: it slides a *target utility*
from an opening down toward its BATNA over the deadline, following the
time-dependent concession functions of Faratin, Sierra & Jennings (1998):

```
target(round) = BATNA + opening · (1 − (round / (deadline−1))^(1/β)) · (ideal − BATNA)
```

This is the **standard** concession-tactic family from the automated-negotiation
literature — the `t^(1/β)` curve with Boulware / Linear / Conceder as its canonical
shapes (Faratin et al. 1998; taxonomy by Baarslag et al.). It is the common
baseline in the ANAC competition and the GENIUS / NegMAS platforms (NegMAS ships
`BoulwareTBNegotiator`, `ConcederTBNegotiator`, etc.), so the arms here are a
direct implementation of those tactics rather than anything bespoke. Background:
[Automated negotiation](https://en.wikipedia.org/wiki/Automated_negotiation),
[Boulwarism](https://en.wikipedia.org/wiki/Boulwarism) (origin of the "Boulware"
name), [BATNA](https://en.wikipedia.org/wiki/Best_alternative_to_a_negotiated_agreement),
[ZOPA](https://en.wikipedia.org/wiki/Zone_of_possible_agreement).

Each round the agent **offers** the most generous outcome that still meets its
target, and **accepts** the standing offer once that offer's utility clears the
target (never below BATNA). Only `opening` and `β` differ between arms:

| Strategy      | opening | β    | Behavior                                                        | Source |
|---------------|:-------:|:----:|-----------------------------------------------------------------|:------:|
| `control`     | 0.85    | 1.0  | Neutral baseline; steady concession, no anchoring.              | —      |
| `boulware`    | 1.00    | 0.05 | Holds near the opening, then concedes sharply at the deadline.  | [1]    |
| `linear`      | 0.90    | 1.0  | Concedes at an approximately constant rate.                     | [1]    |
| `conceder`    | 0.90    | 5.0  | Concedes rapidly and early.                                     | [1]    |
| `tit_for_tat` | 0.85    | —    | Opens cooperatively, then mirrors the opponent's concession.    | [2]    |
| `anchoring`   | 1.00    | 1.0  | Extreme-but-feasible opening anchor, then a fixed linear slope. | [3]    |
| `logrolling`  | 0.90    | 1.0  | Trades across issues toward the Pareto frontier (multi-issue).  | [4]    |

`tit_for_tat` is not time-based: its target reciprocates the opponent's measured
concession on the agent's own utility scale. `logrolling` reuses the linear
schedule but, among offers that meet its target, deterministically prefers the
values the opponent last asked for — conceding on issues it values least without
ever seeing the opponent's private utilities. It is skipped on single-issue
scenarios (nothing to trade).

The **opponents** are the same machinery with fixed parameters (`hardliner` =
opening 1.0/β 0.02, `fast_conceder` = 0.9/6.0, `reciprocal` = tit-for-tat,
`integrative` = 0.9/1.0 logrolling). The agent side is the treatment under test;
the opponent is the environment you sweep over. With `--agent llm`, only the
tested side becomes a language model — the opponent stays scripted. For LLM-vs-LLM
self-play, see **A2A self-play** below.

## A2A self-play (buyer personas vs seller strategies)

`--agent a2a` runs the deferred **agent-to-agent** condition: *both* sides are
language models. It maps directly onto the benchmark's existing axes, so the report
and dashboard work unchanged:

- The **seller** is the scored agent (`target_role = seller`), an `LLMNegotiator`
  steered by one of the seven known **strategies** (the same `STRATEGY_GUIDANCE`) —
  the treatment, `control` included, so Δ-vs-control still applies.
- The **buyer** is the **opponent**, an `LLMNegotiator` steered by a **persona** — a
  behavioral disposition carried on the standard `opponent` axis. Personas vary tone,
  urgency, price-sensitivity, risk tolerance, aggressiveness, and cooperativeness
  only; the buyer's private utilities, BATNA, and ideal come from the scenario and
  stay identical across personas, so scoring and the leakage guarantee are unchanged.
- **The two sides can talk.** Besides its offer, each side may send a short public
  `message` (optional, delivered up to 300 characters; the full text is kept in the
  trace and flagged if cut). The other side sees it in its `history` next to the
  offer. Each move's `rationale` stays private: it goes into the trace for you and is
  never shown to the other agent. Messages are cheap talk: only the offer fields bind,
  and a model may reveal, withhold, or bluff. The leakage guarantee is therefore that
  the harness never inserts private economics into a prompt; what a model chooses to
  say is part of the negotiation. `--agent llm` has no channel, since its scripted
  opponent can't read or reply.
- **Invalid moves are corrected, withheld, and counted.** When a policy constraint
  fires on a move (an off-grid value snapped to the nearest option, or a quote or
  accept below the walk-away raised to it), the corrected offer is what's played, and
  that move's message is withheld from the opponent, since it quotes the invalid
  terms. The agent that made the move is told, in its own history entry, what it
  sent and what was played instead (e.g. `policy: you offered price=240, below your
  minimum of 245; played price=245`), so it can correct itself. For one issue the
  walk-away itself is directional in the payload (`{"price": {"at_most": 160}}` for a
  buyer, `{"at_least": 245}` for a seller), so which side of the number is acceptable
  is stated, not implied. The opponent only
  sees the corrected offer, since the note would reveal the author's walk-away. The
  trace keeps the model's original move and message. `rfq-bench report`
  counts firings by role, kind and arm. The dashboard shows a per-episode `policy`
  column, a `policy/ep` column per arm, and in the replay the constraint and the
  withheld message.

```bash
uv run rfq-bench list                              # includes the persona names
uv run rfq-bench run --agent a2a \
    --strategies control,boulware,anchoring \
    --personas neutral,bargain_hunter,hardball \
    --overwrite -y                                 # writes results/a2a_v0.jsonl
uv run rfq-bench report --traces results/a2a_v0.jsonl
uv run rfq-bench dashboard --traces results/a2a_v0.jsonl

# A run made from a config file can be re-scored / re-rendered from the same file,
# which supplies the traces, data dir, dashboard path and [dashboard] settings:
uv run rfq-bench report    --config configs/a2a-price.toml
uv run rfq-bench dashboard --config configs/a2a-price.toml
```

**Primary personas** (procurement-grounded, the defaults in `configs/a2a-grid.toml`):
`cost_focused`, `total_value`, `reliability_first`, `relationship_focused`, with
`neutral` as the Δ baseline. These are the ones to run for a real measurement — each
maps to a recognized negotiation motive; see [Persona design and scientific basis](#persona-design-and-scientific-basis)
below.

The price-only suite (`configs/a2a-price.toml`) uses a different set, fitted to a
single issue: `neutral`, `cost_focused`, `relationship_focused`, `supply_security`,
`time_pressured`. See [Price-only persona set](#price-only-persona-set-configsa2a-pricetoml).

There are also six **generic built-in personas** — `neutral`, `bargain_hunter`,
`time_pressured`, `relationship_builder`, `hardball`, `risk_averse` — hard-coded in
`agent/personas.py` as short disposition strings (`time_pressured` is overridden by
its file in `prompts/personas/`). They are convenience dispositions
for quick smoke checks (e.g. `configs/a2a-smoke.toml`), not the grounded experiment;
the primary set above (defined in `prompts/personas/*.md`) is merged on top, so all
names are available at once.

The matrix is `scenarios × strategies × personas × first-speakers × seeds` (roles
are pinned — the seller plays the strategy, the buyer the persona — so there is no
role mirror). Because personas sit on the opponent axis, `rfq-bench report` prints
the usual `S_s`/Δ table for seller strategies and the separate tracks, and the
dashboard's leaderboard, Δ forest plot, **strategy × opponent heatmap** (opponent =
persona), and episode replay all populate with no extra flags.

**Both sides are scored, from the same traces.** Both parties are real LLM agents,
so `rfq-bench report` prints a second table for the **buyer**: the buyer's captured
value grouped by persona (Δ vs the `neutral` persona as the baseline). The
dashboard adds a **scored side: seller / buyer** toggle in the leaderboard — flip
it and the leaderboard, forest plot, and heatmap re-group by persona and score the
buyer instead (no rerun; the buyer's `q` is on every trace).

Two notes carry over from the LLM path:

- **Cost.** Every round now makes **two** live API calls (buyer + seller), so cost
  roughly doubles vs `--agent llm`; the run plan warns before it starts, and both
  sides' spend is recorded per role on each trace.
- **Determinism.** Two independent model streams mean seeds and temperature matter
  (byte-identical replay is an offline-only guarantee). A2A writes to its own
  `results/a2a_v0.jsonl` so the immutable offline suite is never mixed in.

The same faithful adjustments apply symmetrically to both sides (snap-to-grid,
shop reservation floor); no scripted-strategy value is ever substituted for either
model, and an unusable reply from **either** side excludes the episode.

### Persona design and scientific basis

Beyond the disposition-only built-ins listed above, the benchmark ships four
**procurement-grounded buyer personas** (the defaults in `configs/a2a-grid.toml`,
defined in `prompts/personas/*.md`). Each maps a recognized negotiation motive or
procurement archetype onto the `neutral`-baselined opponent axis. Like every
persona they steer *behavior only* — tone, aspiration, information sharing, risk
tolerance — while the buyer's private utilities, BATNA, and ideal stay fixed by the
scenario, so `q` remains comparable persona-to-persona and the leakage guarantee
holds.

| Persona (`file`) | Archetype | Scientific basis | Sources |
| --- | --- | --- | --- |
| `cost_focused` | Leverage / Cost-Focused Buyer | A competitive, distributive negotiation motive: high aspiration, strong BATNA, limited information sharing, and emphasis on price extraction. Kelly & Chicksand's systematic review distinguishes adversarial/distributive bargaining from integrative negotiation; Sebenius grounds the reliance on a strong BATNA, reservation thresholds, and willingness to walk away. | Kelly & Chicksand (2024); Sebenius (2017) |
| `total_value` | Integrative / Total-Value Buyer | A collaborative, problem-solving orientation: considers multiple issues, shares relevant preferences, and searches for trade-offs that improve joint outcomes. Kelly & Chicksand's review contrasts integrative with distributive bargaining; Elgoibar et al. link trust and trustworthiness to information sharing, trade-offs, and joint-value creation. | Kelly & Chicksand (2024); Elgoibar et al. (2021) |
| `reliability_first` | Risk- and Compliance-First Buyer | High sensitivity to losses, uncertainty, and failure. The buyer may accept a higher price for delivery certainty, quality guarantees, or regulatory compliance. Prospect Theory provides the behavioral basis for loss sensitivity and certainty preferences; Choudhary et al. cover supply-risk assessment and multi-criteria risk evaluation in procurement contexts. | Kahneman & Tversky (1979); Choudhary et al. (2023) |
| `relationship_focused` | Relationship-Focused / Continuity Buyer | Values reliability, information sharing, cooperation, and future supplier continuity in addition to the current transaction price. Kumar et al. address long-term buyer–supplier relationships, information access, flexibility, sustainability, fairness, and supplier experience; Elgoibar et al. support trust, reciprocity, and cooperative negotiation. | Kumar et al. (2025); Elgoibar et al. (2021) |

#### Price-only persona set (`configs/a2a-price.toml`)

With price as the only issue, a persona defined by a trade-off against other issues
has nothing to trade. `total_value` (price vs the whole package) and
`reliability_first` (price vs delivery and warranty) behaved like `neutral` in a
price-only run, and their Δ confidence intervals included zero. They stay in the
multi-issue grid and are replaced in the price suite:

| Persona (`file`) | Archetype | Basis on a single issue | Sources |
| --- | --- | --- | --- |
| `neutral`, `cost_focused`, `relationship_focused` | as above | Their core is about price or cooperation, not other issues, so it carries over. | as above |
| `supply_security` | Risk-First Buyer, single-issue form | On one issue the risk isn't poor delivery but *not closing the deal* (supply risk). Loss aversion then shows as paying a premium for a certain agreement, within the walk-away. Same basis as `reliability_first`, applied to the risk that exists here. | Kahneman & Tversky (1979); Choudhary et al. (2023) |
| `time_pressured` | Time-Pressured Buyer | Delay is costly: it prefers a quick acceptable deal over the best price, and grows more willing to pay near the deadline. Experimental research shows that time pressure interacts with negotiation orientation and can either increase competitiveness or encourage faster cooperation. Defined in `prompts/personas/time_pressured.md`, overriding the generic built-in of the same name. | Carnevale & Lawler (1986) |

**References** (freely available unless marked)

- Carnevale, P. J. D., & Lawler, E. J. (1986). *Time pressure and the development of integrative agreements in bilateral negotiations.* Journal of Conflict Resolution, 30(4), 636–659. (Free availability not verified.)
- Choudhary, N. A., et al. (2023). *Risk assessment in supply chains.* (Open access via PMC.)
- Elgoibar, P., Munduate, L., & Euwema, M. (2021). *Increasing integrative negotiation through trustworthiness and trust.*
- Kahneman, D., & Tversky, A. (1979). *Prospect theory: An analysis of decision under risk.* Econometrica, 47(2), 263–291. (MIT-hosted PDF.)
- Kelly, S., & Chicksand, D. (2024). *A critical exploration of bargaining in purchasing and supply management.* (Open-access systematic review.)
- Kumar, N., et al. (2025). *Managing buyer experience in a buyer–supplier relationship.* (Open-access PDF.)
- Sebenius, J. K. (2017). *BATNAs in negotiation: Common errors and three kinds of "no".* (Harvard DASH PDF.)

## Laya decision agent (`--agent laya`)

An experimental third kind of agent: **Laya**, a small, local, **non-generative**
decision model (typed `choice`/yes-no answers, ~20–35 ms/turn, deterministic, no API
cost). It decides the move via typed questions in one forward pass: **Laya picks each
issue's value itself** (one `choice` per issue → the offered package) and owns accept/walk
(`noul`). There is no code concession schedule — the offer is Laya's. Two shop rails keep
moves legal (a below-floor accept is refused; a below-floor pick is raised to the floor).
Because "which price do you propose" is a "what-to-do" question, expect Laya to **anchor**
(as seller it tends to match the price already on the table rather than hold out) — that's
a measurement, not a bug. Full rationale in
[`docs/laya-agent-concept.md`](./docs/laya-agent-concept.md).

It needs a **local Laya server** (English checkpoint) reachable over loopback HTTP
(`POST /predict`). Point at it with `RFQ_BENCH_LAYA_URL` in `.env` if it is not on
`http://127.0.0.1:8000`; thresholds are `RFQ_BENCH_LAYA_ACCEPT_THRESHOLD` /
`RFQ_BENCH_LAYA_WALK_THRESHOLD` (choose them on labelled data).

```bash
uv run rfq-bench run --config configs/laya-smoke.toml    # Laya as seller vs scripted buyers
uv run rfq-bench report    --traces results/laya_v0.jsonl --control laya
uv run rfq-bench dashboard --traces results/laya_v0.jsonl
```

Laya plays the seller; each **strategy is injected as guidance text Laya reads** (like the
LLM agent), so the seven strategies form arms that can bias Laya's own offer choice:
`report` shows `S_s`/Δ per strategy with `control` as the baseline, comparable to the
scripted and LLM versions of the same strategy (expect weak differentiation — whether Laya
acts on the words is exactly what this measures). Use `configs/laya-full.toml` for all
strategies × all scripted buyers. It is a
**separate agent condition** written to `results/laya_v0.jsonl`;
compare it to `llm`/`scripted` side by side, never pooled. Traces record `token_cost`
(Laya's input tokens) with no USD cost (local). Being a deterministic forward pass, its
traces are reproducible. Known limitations (hybrid attribution, ordinal weakness,
verbalization sensitivity) and the planned cascade experiment are in the concept doc.

## How an episode is simulated (offline)

Everything runs deterministically on a NegMAS **SAOMechanism** — no randomness,
so the same inputs reproduce byte-identical traces.

1. **Build the game.** `Scenario.issues` become NegMAS issues; the outcome space
   is their Cartesian product. Each party gets an exact weighted-additive utility
   function with `reserved_value` set to its BATNA.
2. **Wire the negotiators.** The tested agent (a strategy policy, or the
   `LLMNegotiator`) and the opponent policy are each wrapped in a
   `PolicyNegotiator` and added to the mechanism in first-speaker order.
3. **Run.** NegMAS drives alternating offers/counter-offers until an acceptance,
   a walk-away (`END_NEGOTIATION`), or the deadline. Each negotiator sees only its
   own preferences plus the public offer history.
4. **Record.** The offer sequence, agreement (or no-deal), realized utilities,
   validity flags, latency, and token cost are written to an **immutable JSONL
   trace**. Scoring reads that trace and never re-runs the model.

The full run is a **balanced matrix**: every cell is mirrored across
`--roles {buyer,seller}` (which side is the tested agent) and
`--first-speakers {buyer,seller}` (who opens), repeated over `--seeds`, and swept
over `--strategies` × `--opponents` × scenarios. Note: scripted policies have no
stochastic branch, so extra `--seeds` currently produce identical episodes; seeds
matter once a nonzero-temperature LLM or stochastic opponent is in play.

## Metrics tracked

Two tracks, kept strictly separate — the strategy score is **never** blended with
the quality/safety/cost metrics.

**The strategy score (the causal comparison).**

- **`q` (per episode)** — `clip((U − d) / (I − d), 0, 1)`: value the agent
  captured above its BATNA `d`, relative to its ideal `I`. A no-deal falls back to
  the BATNA, so `q = 0`. If `I == d` the scenario is **degenerate** and excluded
  (not scored 0).
- **`S_s` (per strategy)** — `100 × macro-average of q` over all matched cells
  (every scenario × opponent × role × first-speaker × seed weighted equally).
- **`Δ_s` = `S_s − S_control`** — the effect vs the identical control condition,
  with a **95% CI bootstrapped by resampling scenarios (clustered)**. For an arm
  that only runs on a subset (e.g. `logrolling`), control is restricted to the
  same scenarios so the point estimate and CI share one basis.

**Separate tracks (reported alongside, never folded in).**

- **`agree%`** — fraction of episodes that reached agreement.
- **`joint`** — mean joint surplus of agreements: `(U_buyer − d_buyer) +
  (U_seller − d_seller)`, i.e. total value created above both BATNAs.
- **`pareto%`** — share of agreements on the Pareto frontier (no other outcome
  dominates them for both parties). **Higher is better** (`100%` = every deal is
  Pareto-efficient). The interactive dashboard computes the same value.
- **`rounds`** — mean rounds-to-close (a speed/effort proxy).
- **`walk✓`** — walk-away accuracy: on no-ZOPA scenarios, the share of episodes
  that correctly reached no deal instead of accepting below BATNA.
- **latency, tokens & cost** — mean wall-clock and (LLM-mode) token usage per
  episode. When the provider reports real spend (e.g. **OpenRouter**'s
  `usage.cost`), the actual **USD cost** is captured on each trace and shown as
  `$/ep` in `report` and per-episode in the dashboard; providers that don't
  report cost show "not reported" rather than a guess.

**Strategy fidelity (manipulation check).** An LLM or Laya agent only *reads* its
strategy as guidance text, so it may not play it. `report` replays each arm's
registry definition (opening, concession shape `β`, acceptance rule) over the
agent's actual episode history and compares turn by turn
(`report/fidelity.py`):

- **`open` / `bias` / `|gap|`** — agent offer − the offer the strategy would make,
  as a share of the agent's BATNA→ideal range (first offer / signed mean / mean
  absolute). `+` = tougher than the strategy, `−` = conceded more.
- **`on-tgt`** — share of offers within ±0.10 of the strategy's offer.
- **`early✗` / `missed✗`** — accepts the strategy would not make / accepts it
  would have made (countered or walked instead), per accept decision.
- **`β fit` / `β ref`** — median concession shape fitted to the agent's offers vs
  the same fit on the strategy's offers (`<1` holds then concedes late, `>1`
  concedes early). Needs ≥ 3 offers per episode.

Scripted agents replay with zero gap by construction (the test anchor). If arms
show a Δ but near-identical fidelity profiles, the model is not differentiating
the strategies. Traces record their effective `deadline` (older traces fall back
to the scenario's). Replays use the *current* strategy code, so traces written
before a strategy's definition changed will show gaps against the new definition.

Per-episode **validity flags** are also stored on each trace:
`agreement_reached`, `legal_agreement`, `no_batna_violation`, and
`correct_walk_away`.

`rfq-bench report` prints the score table plus the separate tracks; `rfq-bench
dashboard` renders the same numbers interactively (leaderboard, Δ forest plot,
strategy×cell heatmap, and a round-by-round replay against the ZOPA band).

## Reading the results

Scales and how to interpret actual values. Some metrics are natural 0–100 scales
with absolute meaning; others (`joint`, `rounds`, `cost`) are best read
**relatively** — compare strategies to each other and to control, not to a fixed
threshold.

| Metric | Scale / range | Direction | How to read a value |
|---|---|---|---|
| `S_s` | 0–100 | higher = agent captured more | `q×100` = % of available surplus above BATNA captured. `0` = stuck at its walk-away; `100` = its ideal; **~50 ≈ an even split** of the surplus. Its meaning comes from the gap to control (Δ). |
| `Δ` | points on the 0–100 scale (realistically ±0–20) | higher = beats control | `S_s − S_control`. `+5` = 5 more surplus-points than neutral negotiating; `0` = same; negative = worse. |
| `95% CI` | points, same units | — | Excludes 0 (e.g. `[+3,+9]`) → a real effect. Straddles 0 (e.g. `[−2,+6]`) → indistinguishable from control. **This gates whether Δ means anything.** |
| `agree%` | 0–100% | context-dependent | Deal rate. The matrix includes the no-ZOPA case where the *right* answer is no-deal, so even a perfect agent won't hit 100%. Read with `S_s`. |
| `joint` | ~0 to ~1.5 (sum of both sides' surplus) | higher = more efficient | Near-constant on single-issue (zero-sum) — ignore it there. On multi-issue it moves: higher = better issue-trading (logrolling). Compare across strategies. |
| `pareto%` | 0–100% | higher = better | Share of agreements on the Pareto frontier. `100%` = every deal is Pareto-efficient; low = value left on the table. |
| `rounds` | 0 to ~2×deadline | lower = faster/cheaper | NegMAS logs ~2 actions/round, so it roughly doubles the deadline. No absolute "good" — compare arms; high = drags out (Boulware) and costs more tokens. |
| `walk✓` | 0–100% | higher = safer | On no-ZOPA cases, % correctly refused. `100` = never signs a losing deal; `—` = no such cases in the filter. |
| `cost` / `tokens` | USD / count, absolute | lower = cheaper | LLM runs only; compare arms. |

**Anchors that matter most:**

- **`S_s ≈ 50` is the "fair split" line** — below it the agent captures *less* than
  half the available surplus on average; above it, more than its even share. (This
  is also why the dashboard heatmap shows no green below 50: green = capturing more
  than half.)
- **Δ + CI is the verdict, not `S_s` alone.** `S_s=38` isn't "bad" if control is
  `35` and the CI on `+3` clears 0 — that's a small but real gain. `S_s=45` with
  control at `48` is a *loss*.
- **`agree%`, `walk✓`, `pareto%`** are absolute 0–100 — judge them on their own.
  **`joint`, `rounds`, `cost`** are comparative — the *difference* between
  strategies is the signal.
- **Check the exclusions first.** Degenerate scenarios and errored LLM episodes are
  dropped (not scored 0), so a run with many errors has a small `n` and shaky
  numbers — trust a comparison only when `n` is healthy.

*Worked example* (from a scripted run): `boulware S_s=8, Δ=−14.8, CI=[−19.7,−7.3],
agree%=20%` reads as *"Boulware captured far less than neutral — a real,
significant loss (CI well below 0) — because it walked away 80% of the time holding
out for a deal it rarely got."* Everything lines up: low score, significant
negative Δ, low agreement.

## Limitations & known drawbacks

Read this before drawing conclusions. The benchmark is strong on **internal
validity and auditability** (frozen dimensions, immutable traces, scoring outside
the engine) and deliberately weak on **external validity and statistical power** at
v0.1. Rule of thumb: read **Δ + CI, not `S_s` alone**; always check the exclusions;
treat A2A numbers as exploratory.

**Design tradeoffs (by construction, not bugs).**

- **Within-agent, not a model leaderboard.** Only the strategy varies; everything
  else is frozen. This buys clean causal attribution but means you cannot compare
  models, prompts, or providers with it.
- **Measures one thing: value captured above BATNA.** `S_s`/`q` ignore fairness,
  joint surplus, efficiency, safety, speed, and cost — those are separate tracks,
  never folded in. A high `S_s` can just mean **exploiting a weak opponent**, not
  skill — especially on single-issue scenarios, which are zero-sum (`joint` is
  ~constant and uninformative there).
- **Closed, static world.** Bilateral SAO, fixed issue set, exact weighted-additive
  utilities, outcome space ≤27. No dynamic issues, outside options, multi-party, or
  nonlinear/interacting preferences.

**Known bugs / mislabeled metrics.**

- **Extra `--seeds` are no-ops for scripted policies** (no stochastic branch → identical
  episodes), which inflates `n` with fake precision. Seeds matter only with an LLM or
  stochastic side.

**Statistical limits.**

- **Tiny anchor set (5 scenarios).** CIs cluster by scenario, so with a handful of
  clusters the intervals are coarse and unstable — easy to over-/under-claim
  significance. The 50-/500-scenario milestones are future work; today's numbers are
  directional.
- **Macro-average weights every cell equally** regardless of `n`; small cells are noisy.
- **Selection effects.** Degenerate scenarios and errored LLM episodes are dropped, so a
  model that fails on hard cases gets a smaller, easier `n`. Check the counts first.

**LLM mode.**

- Snap-to-grid and the reservation-floor clamp **alter the model's raw move** (logged,
  rare, not scripted substitutions).
- **Nondeterminism** — byte-identical replay is offline-only; LLM runs depend on the
  decoding snapshot and provider behavior.
- **Provider/tool-calling dependence** — reasoning models often return the move as
  message content rather than tool arguments (recovered by a content-JSON fallback) and
  can silently truncate if `RFQ_BENCH_MAX_TOKENS` is too low. Real USD cost only appears
  when the provider reports `usage.cost`.
- `--max-rounds` cuts cost but **changes outcomes** (capped runs are not comparable to
  full-deadline runs).
- **System-prompt provenance gap.** The system prompt is configurable via markdown, but
  only the model/decoding config is snapshotted into each trace (`LLMConfig`) — the
  resolved prompt *text* is not. Record which prompt file you used alongside a run's
  results, especially when comparing custom prompts.

**A2A self-play.**

- **Prompt-guided ≠ provably distinct.** Offline scripted strategies are guaranteed
  behaviorally distinct; in A2A both the seller strategy and the buyer persona are
  one-line prompt steers the model may blur or ignore. Nothing verifies that a persona
  or strategy actually changed behavior.
- **Single-factor machinery on a two-factor design.** Persona maps onto the `opponent`
  axis and reuses the offline Δ pipeline, so there is **no persona×strategy interaction
  term**, and CIs still cluster only by scenario.
- **Arbitrary baselines** — seller Δ vs `control`, buyer Δ vs `neutral` (a convention,
  not a validated zero).
- **Same model on both sides** (one `RFQ_BENCH_MODEL`) → correlated behavior, not a
  diverse market; **no cross-play** (which model plays which role is fixed).
- **~2× cost and full nondeterminism**; seeds matter and results are noisier.
- **Persona "framing only, fixed economics" is a convention** enforced by a leakage test
  screening the guidance text, not by the type system.

**Scope (deferred, designed-for but not built):** `ShopRunner`/Shopware quote adapter,
A2A cross-play, additional model families, the 500-scenario target.

## References

1. Faratin, Sierra & Jennings (1998), *Negotiation decision functions for
   autonomous agents.*
2. Baarslag, Hindriks & Jonker (2013), *A Tit for Tat Negotiation Strategy for
   Real-Time Bilateral Negotiations.*
3. Galinsky & Mussweiler (2001), *First offers as anchors.*
4. Tajima & Fraser (2001), *Logrolling Procedure for Multi-Issue Negotiation.*

## Layout

`src/rfq_bench/core` is the deterministic scoring kernel (no NegMAS, no network).
`strategies/` and `opponents/` are concession policies; `agent/` is the fixed LLM
adapter. **NegMAS is the negotiation engine**: `runners/negmas_engine.py` maps our
issues/utilities/policies onto a NegMAS `SAOMechanism` and `SAONegotiator`, and
`runners/offline.py` runs the matrix and assembles the immutable trace. `report/`
aggregates. Scenarios live in `data/scenarios/` — the v0.1 anchor set is immutable.

## License

MIT.
