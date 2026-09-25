# rfq-bench — Agent Build Guide

A reproducible **within-agent negotiation benchmark** for buyer/seller language agents
in e-commerce. One fixed LLM agent negotiates against deterministic opponents in
controlled bilateral SAO sessions; **only the strategy policy varies**. The benchmark
measures how much value each strategy captures above the agent's BATNA.

This document is the source of truth for building the project. It is written for coding
agents. Follow it exactly; where it says "do not," treat it as a hard constraint that
protects scientific validity.

---

## 1. Scientific contract (read first, do not violate)

The experiment is a **controlled within-agent comparison**, not a model leaderboard.

- **Fixed across every run (frozen):** model version, role prompt, tools, decoding
  settings (temperature, top-p, max tokens, seed where supported), private-information
  exposure, action schema, and the outcome-space version.
- **The only treatment:** the `strategy` policy.
- **Scored artifact:** an immutable trace. Scoring happens **outside** NegMAS, on the
  persisted trace, so it can be recomputed and audited without re-running the LLM.

If a change would alter any frozen dimension, it must be a new, explicitly versioned
benchmark condition — never an in-place edit to an existing arm. **Keep the original
deterministic offline suite immutable**; add scenarios/opponents/conditions rather than
replacing them, so later results stay comparable.

### The score

For target party *i* in episode *e*, with realized utility `U`, BATNA utility `d`, and
ideal attainable utility `I`:

```
q_i,e = clip( (U_i,e - d_i) / (I_i - d_i), 0, 1 )
```

- No agreement → outcome is the BATNA → normally `q = 0`.
- If `I_i == d_i`, mark the scenario **degenerate** (do not divide by zero).
- No-ZOPA refusal is **not** scored as 0; it is reported separately as a safety result.

Strategy score (macro-average, every cell equally weighted over scenarios × opponents ×
roles × first-speaker order × repetitions):

```
S_s = 100 * (1/N_s) * Σ q_i,e
```

Strategy effect vs. the identical control condition, with a confidence interval
**clustered by scenario**:

```
Δ_s = S_s - S_control
```

The score measures only captured-value-above-BATNA. It does **not** measure fairness,
joint surplus, PO/constraint validity, walk-away safety, speed, cost, robustness, or
language quality. Compute and report those **separately** (see §8).

---

## 2. Scope for this build

Two milestones are in scope. Do not build beyond them without being asked.

- **v0.1 — Offline benchmark (primary).** `OfflineRunner` + shared core + deterministic
  scorer. Bilateral SAO, closed issue set, scripted opponents, fixed seeds.
- **Iteration 1 — Smoke suite.** The 50-scenario validation milestone. Harden scenario
  generation, hidden economics, ZOPA calculation, Pareto/PO validation, deterministic
  metrics, and the failure matrix.
- **A2A self-play (implemented).** `--agent a2a`: LLM seller **strategy** (the scored
  treatment, `target_role='seller'`, `control` included) vs LLM buyer **persona** (the
  opponent). A **new versioned condition** written to its own `results/a2a_v0.jsonl`.
  Personas map onto the standard **opponent** axis, so it reuses the SAO engine, the
  `LLMNegotiator`, the seller strategies, the pure scorer, and the report/dashboard
  unchanged. The buyer persona is prompt-level disposition only (fixed economics), so
  it touches no frozen offline dimension and the v0.1 offline suite stays immutable.
  See README "A2A self-play". Cross-play is the separate `--agent duel` condition
  (below); additional model families remain out of scope.

- **Laya decision agent (implemented).** `--agent laya`: a local, non-generative
  decision model (Laya) as the tested agent, over loopback HTTP to a local server
  (English checkpoint). **Laya decides the move**: one `choice` per issue picks the offer,
  and `accept`/`reachable` (`noul`) gate accept/walk — no code concession schedule. The
  strategy is injected as guidance text Laya reads. Two legality rails only (below-floor
  accept refused; below-floor pick raised to floor). A new versioned condition in
  `results/laya_v0.jsonl`; reuses the SAO engine, contracts, and scorer unchanged;
  deterministic forward pass (reproducible traces). Known cost: "which price do you
  propose" invites anchoring — a measurement. Concept and eval plan (incl. the deferred
  cascade experiment) in `docs/laya-agent-concept.md`. The offline suite stays immutable.

- **Model duel (implemented).** `--agent duel --models A,B`: two models negotiate
  against each other in both roles (`cross`: A→B and B→A; `full`: plus both
  self-play pairs). The **model** is the treatment; strategy/persona held at
  `control`/`neutral`, decoding/prompts/tool_choice shared. A **new versioned
  condition** (`mode = "duel"`, own `results/duel_v0.jsonl`, per-side model in
  `Trace.models_by_role`). Scored by `report/duel.py`: per cell, a model's score is
  the mean of its seller q and buyer q against the other model (role advantage
  cancels), Δ with scenario-clustered CI; role skills with `full`. Reuses the SAO
  engine, `LLMNegotiator`, scorer, and compare template. This is the "cross-play"
  previously listed as out of scope; additional model families remain out of scope.

- **Cross-run comparison (implemented).** `rfq-bench compare A.jsonl B.jsonl`: a
  report over two existing trace files (e.g. the same condition with two models).
  Pure, over persisted traces; pairs matched cells (scenario × strategy × opponent ×
  role × first speaker), reports `S_B − S_A` with a scenario-clustered CI, and warns
  on any non-model condition difference. It is a between-run readout, **not** the
  within-agent strategy effect Δ_s, and never pools runs. It adds no runner, model
  family, or condition. See README "Comparing two runs".

**Out of scope now (design for, do not implement):** `ShopRunner` / Shopware quote
adapter, the Laya cascade (`--agent cascade`) and the alternative Laya
move designs (package-choice, gates-only), threshold calibration on ground-truth labels,
additional model families, the 500-scenario PM target. Keep contracts platform-neutral
so these attach later without touching the core.

---

## 3. Tech stack & tooling

- **Language:** Python (NegMAS is Python). Target **3.11+**.
- **Env & deps:** [`uv`](https://docs.astral.sh/uv/). Use `uv` for the venv and all
  dependency management; a `pyproject.toml` is the single manifest. Never edit an
  ad-hoc `requirements.txt` by hand.
- **Negotiation engine:** [`negmas`](https://negmas.readthedocs.io/) — supplies the SAO
  mechanism, utility evaluation, deadlines, and negotiator lifecycle.
- **LLM client:** any **OpenAI-compatible API**. The client MUST read a **configurable
  base URL**, model name, and API key from config/env — never hardcode a provider or
  endpoint. See §6.
- **Lint/format:** `ruff` (lint + format).
- **Types:** `mypy` (or `pyright`), strict on the core package.
- **Tests:** `pytest`.
- **Config:** `pydantic` (v2) models + `pydantic-settings` for env-driven settings;
  scenarios/datasets in JSON or CSV.

### Common commands

```bash
uv sync                      # create/refresh the environment from pyproject + lockfile
uv run rfq-bench --help      # CLI entry point
uv run pytest                # run tests
uv run pytest -m "not llm"   # skip tests that call a live LLM
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Add the CLI as a project script in `pyproject.toml` (`[project.scripts] rfq-bench = ...`).

---

## 4. Repository layout

```
rfq-bench/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example                 # OPENAI_BASE_URL, OPENAI_API_KEY, MODEL, etc.
├── src/rfq_bench/
│   ├── __init__.py
│   ├── cli.py                    # `rfq-bench run|score|report`
│   ├── core/                     # SHARED CORE — offline and (future) shop reuse this
│   │   ├── contracts.py          # pydantic schemas: Scenario, Issue, Outcome, Trace…
│   │   ├── outcome_space.py      # closed issue set, legal values, versioned
│   │   ├── utility.py            # utility fns, BATNA / reservation value, ideal point
│   │   ├── zopa.py               # ZOPA + degeneracy detection
│   │   └── scoring.py            # q_i,e, S_s, Δ_s — pure functions over immutable traces
│   ├── strategies/               # the ONLY treatment
│   │   ├── base.py               # StrategyPolicy protocol
│   │   ├── control.py boulware.py linear.py conceder.py
│   │   ├── tit_for_tat.py anchoring.py logrolling.py
│   ├── opponents/                # deterministic, seeded scripted policies
│   │   ├── base.py hardliner.py fast_conceder.py reciprocal.py integrative.py
│   ├── agent/
│   │   ├── negotiator.py         # NegMAS SAONegotiator adapter wrapping the LLM
│   │   ├── llm_client.py         # OpenAI-compatible client, configurable base URL
│   │   └── prompts.py            # FROZEN role prompt + action-schema instructions
│   ├── runners/
│   │   ├── offline.py            # OfflineRunner: run the matrix, assemble the trace
│   │   └── negmas_engine.py      # NegMAS bindings: outcome space, ufuns, PolicyNegotiator
│   ├── datasets/                 # scenario loaders (JSON/CSV) + validators
│   └── report/                   # aggregation, CIs clustered by scenario, tables
├── data/scenarios/               # committed scenario datasets (immutable anchor sets)
├── results/                      # run outputs (git-ignored); traces are immutable once written
└── tests/
```

Rules:
- `core/` must not import from `runners/`, `agent/`, or any LLM/NegMAS-runtime code
  paths that would make scoring non-deterministic. Scoring is pure and offline.
- Strategies and opponents depend only on `core/` contracts.

---

## 5. Data & domain model (`core/contracts.py`)

Model these as pydantic types. Keep everything the LLM must **not** see marked private.

- **Scenario:** id, kind (`single_issue` | `multi_issue` | `no_zopa_diagnostic`),
  product, issues, outcome-space version, deadline (rounds), seed.
- **Issue:** name (e.g. `price`, `delivery`, `warranty`), legal values / range, and per-
  party weight. No dynamic issue discovery.
- **Private economics (never sent to the model):** buyer value, seller cost, BATNAs,
  reservation values, ideal points, approval limits.
- **Outcome / Agreement:** the accepted contract terms; scorable by the utility engine.
- **Trace:** transcript, all offers, final agreement (or no-deal), realized utilities,
  strategy id, opponent id, role, first-speaker order, seed, full model config, latency,
  token cost, validity flags, and provenance (code version, run start, fingerprints of
  every model-visible text; serving provider per LLM step). The trace is
  **append-only and immutable** once a run completes.

Scenario dataset shape (`data/scenarios/*.json` or `.csv`) defines: products, buyer
value, seller cost, BATNAs, issue weights, constraints, and deadlines.

### Scenario matrix (v0.1)

- 2 single-issue price cases.
- 2 multi-issue cases (price + delivery + warranty).
- 1 separate no-ZOPA diagnostic (tests correct walk-away).
- **Balance:** every cell mirrored across buyer/seller role **and** first-speaker order,
  repeated with recorded seeds.

---

## 6. LLM agent (fixed) & OpenAI-compatible client

- `agent/llm_client.py` targets an **OpenAI-compatible** chat/completions API. It reads
  **base URL, model, api key** (and optional org/headers) from settings. Provide these
  via env; document them in `.env.example`:

  ```
  OPENAI_BASE_URL=https://<any-openai-compatible-endpoint>/v1
  OPENAI_API_KEY=...
  RFQ_BENCH_MODEL=<model-id>
  RFQ_BENCH_TEMPERATURE=0
  RFQ_BENCH_MAX_TOKENS=...
  RFQ_BENCH_SEED=...        # forwarded when the endpoint supports it
  ```

- The **role prompt, tools, and decoding config are frozen** and identical across every
  strategy arm. Record the exact resolved config into each trace so a run is
  reproducible from the trace alone.
- `agent/negotiator.py` is a NegMAS `SAONegotiator` (custom adapter). It receives the
  canonical state, asks the LLM for one action, and returns **one validated** offer /
  accept / terminate. Treat the LLM purely as a language/action adapter over the
  structured negotiation state.
- Strategy is injected into the negotiator, not baked into the prompt behavior in a way
  that changes any frozen dimension. Only the strategy policy differs between matched
  runs.

---

## 7. Strategies (the treatment) & opponents

Each strategy MUST specify, explicitly and testably: **opening offer, concession
schedule, acceptance rule, tie-breaking rule, and any use of opponent information.**

| Strategy    | Operational definition | Source |
|-------------|------------------------|--------|
| Control     | Neutral utility-target baseline; no concession/anchoring intervention. | Baseline |
| Boulware    | Hold the initial target for most of the session, concede sharply near the deadline. | [1] |
| Linear      | Concede at ~constant rate from opening target toward the reservation value. | [1] |
| Conceder    | Move rapidly toward the reservation value early. | [1] |
| Tit-for-Tat | Open cooperatively, then reciprocate the opponent's concession/movement, on the agent's own utility scale. | [2] |
| Anchoring   | Ambitious-but-feasible first offer; keep later behavior fixed to isolate the opening anchor. | [3] |
| Logrolling  | Trade issues of differing priority; propose packages toward the Pareto frontier. **Multi-issue only.** | [4] |

Deterministic opponents (scripted, fixed-seed): **Hardliner, Fast conceder, Reciprocal
negotiator, Integrative preference-trading bot.** LLM opponents are optional and only for
external validity — not part of the v0 causal comparison.

---

## 8. Runner, scoring & reporting

**OfflineRunner** (`runners/offline.py`):
1. Instantiate a NegMAS `SAOMechanism` with the scenario's utility functions and deadline.
2. Attach the fixed LLM negotiator (with the arm's strategy) and the scripted opponent
   (with the fixed seed).
3. Run the session to agreement / no-deal / termination.
4. Persist the immutable trace (see §5) plus latency, token cost, and validity flags.

**Scoring** (`core/scoring.py`) runs on persisted traces, independently of NegMAS:
compute `q_i,e`, then `S_s`, then `Δ_s` vs. control, with scenario-clustered CIs.

**Report separately (never fold into the strategy score):**
- Outcome quality: joint surplus, Pareto efficiency, fairness / buyer-vs-seller split.
- Validity & safety: agreement rate, structured-PO & constraint violations, BATNA
  violations, correct no-ZOPA walk-aways.
- Operational: rounds-to-close, latency, token/API cost, robustness across seeds, trace
  and language quality.

---

## 9. Determinism, reproducibility & validity guardrails

- **Seed everything** the benchmark controls (opponent policy, scenario sampling, any
  tie-breaking). Forward the model seed when the endpoint supports it; record it either
  way.
- **Provenance.** Every trace records `provenance` (git commit, run start, sha256
  fingerprints of each model-visible text: system prompt per role, instruction per
  arm, move tool schema) and each LLM step its serving `provider`. Compare/duel warn
  when fingerprints differ. Pin decoding in the config's `[llm]` table, not `.env`.
  Keepable runs are archived under `data/runs/` (results/ is git-ignored).
- **Traces are immutable.** Writing a trace is append-only; never mutate `results/`
  outputs. Re-scoring reads traces, it does not re-run them.
- **No leakage:** private economics (values, costs, BATNAs, margins, approval limits)
  never enter the prompt or any model-visible payload. Add a test that asserts this.
- **Degenerate scenarios** (`I == d`) are flagged, not scored.
- **Do not pool** offline scores with any future connected-mode scores without a mode
  label.
- Any change to a frozen dimension = a new versioned condition, not an edit.

### Known limitations (keep honest; see README "Limitations & known drawbacks")

- **Scope of the score:** `S_s`/`q` measure only value-capture-above-BATNA — never
  fairness, joint surplus, safety, speed, or cost. A high `S_s` may be opponent
  exploitation, not skill (single-issue scenarios are zero-sum).
- **Seeds are no-ops for scripted policies** (deterministic → identical episodes); they
  matter only with an LLM/stochastic side.
- **Low power at v0.1:** the 5-scenario anchor set makes scenario-clustered CIs coarse;
  macro-averaging weights small cells equally; exclusions (degenerate/errored) shrink and
  can bias `n`. Read Δ+CI and check exclusions, not `S_s` alone.
- **A2A self-play** treats the seller strategy and buyer persona as **one-line prompt
  steers** (not provably-distinct policies), maps persona onto the `opponent` axis
  (single-factor Δ machinery — no persona×strategy interaction term), scores the buyer
  vs a `neutral` baseline, and runs the **same model on both sides** with no cross-play;
  it is ~2× cost, nondeterministic, and exploratory.

---

- **Model duel** scores the pair of models on one setup — not a general ranking. It
  inherits A2A's limits (prompt-steer personas/strategies, nondeterminism, ~2× cost)
  and, on the price suite, has only 6 scored scenarios, so CIs are wide.

## 10. Build order (checklist)

Work top-down; each step should land with tests before the next.

1. Project skeleton: `pyproject.toml` (uv), package layout, CLI stub, `.env.example`,
   ruff/mypy/pytest config.
2. `core/contracts.py` + `outcome_space.py` (versioned closed issue set).
3. `core/utility.py`, `core/zopa.py`, `core/scoring.py` — pure, fully unit-tested,
   including the worked example (`U=80, d=50, I=100 → q=0.60`) and the degenerate case.
4. Deterministic opponents + `strategies/` (start with Control, then time-dependent
   arms, then Tit-for-Tat / Anchoring / Logrolling).
5. `agent/llm_client.py` (OpenAI-compatible, configurable base URL) + `negotiator.py` +
   frozen `prompts.py`.
6. `runners/offline.py` + trace persistence.
7. `datasets/` loaders/validators + the v0.1 scenario matrix in `data/scenarios/`.
8. `report/` aggregation with scenario-clustered CIs; `rfq-bench report`.
9. **Iteration 1 smoke suite:** scale to the 50-scenario validation milestone; add the
   failure matrix and harden generation, hidden economics, ZOPA, PO validation, and
   deterministic metrics.

---

## 11. Testing conventions

- Mark tests that hit a live LLM with `@pytest.mark.llm`; the default suite
  (`uv run pytest -m "not llm"`) must pass with **no** network/API access.
- `core/` (utility, ZOPA, scoring) must be deterministic and covered by exact-value unit
  tests — this is the scientific kernel.
- Add a leakage test asserting no private economics appear in any model-visible payload.
- Snapshot/replay tests: given a recorded trace, scoring must reproduce identical numbers.

---

## 12. Future-proofing (design for, don't build yet)

Keep the scenario/quote schema, outcome space, utility/BATNA, strategy policy, LLM
adapter, scorer, and trace schema **reusable across two runners**. A future `ShopRunner`
maps Shopware quote-created/updated events to the same canonical state (quote id/version,
line items, price, quantity, delivery, warranty, expiry, public history, allowed
actions), runs the same negotiator, and returns one validated action — first read-only /
shadow against historical quotes, then human-approved live counteroffers. Do not let this
future need leak concrete shop/HTTP concerns into `core/` now.

---

## References

1. Faratin, Sierra & Jennings (1998). *Negotiation decision functions for autonomous
   agents.* Robotics and Autonomous Systems 24, 159–182.
2. Baarslag, Hindriks & Jonker (2013). *A Tit for Tat Negotiation Strategy for Real-Time
   Bilateral Negotiations.*
3. Galinsky & Mussweiler (2001). *First offers as anchors.* JPSP 81, 657–669.
4. Tajima & Fraser (2001). *Logrolling Procedure for Multi-Issue Negotiation.* Group
   Decision and Negotiation 10, 217–235.
- NegMAS docs: https://negmas.readthedocs.io/en/latest/readme.html · SAO API:
  https://negmas.readthedocs.io/en/stable/modules/sao.html · Mohammad, Nakadai &
  Greenwald (2021), *NegMAS: A Platform for Automated Negotiations*,
  https://doi.org/10.1007/978-3-030-69322-0_23.
