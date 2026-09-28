# rfq-bench

[![CI](https://github.com/agentic-commerce-lab/rfq-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/agentic-commerce-lab/rfq-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**A reproducible benchmark for AI agents that negotiate B2B deals.**

rfq-bench puts a buyer or seller agent into controlled price and contract negotiations
(think requests for quotation) and measures how much value it secures above its
walk-away point. It is built as a measurement tool: every run is a fixed, recorded
condition, only one thing varies at a time, and all scoring happens on immutable traces
that you can re-score and audit without calling a model again.

Use it to answer questions like:

- Does telling an LLM to play **Boulware** or to **anchor high** actually capture more
  value than a neutral approach?
- Which of two models is the better negotiator when they face **each other**, as buyer
  and as seller?
- Does a buyer **persona** (cost-focused, time-pressured, relationship-focused) change the
  deal a seller agent gets?
- What does each approach **cost** in tokens, money and time?

## Capabilities

- **Five agent conditions.** Deterministic strategy policies (`scripted`), one LLM against
  scripted opponents (`llm`), LLM-vs-LLM self-play with seller strategies and buyer
  personas (`a2a`), head-to-head **model duels** in both roles (`duel`), and an
  experimental local decision model (`laya`).
- **Literature-grounded arms.** Seven seller strategies (control, Boulware, linear,
  conceder, tit-for-tat, anchoring, logrolling), four scripted opponents, and buyer
  personas grounded in procurement research.
- **A score with a clear meaning.** The share of the available gain captured above the
  walk-away, macro-averaged over a balanced matrix, with effects and 95% CIs clustered by
  scenario. Deal rate, joint surplus, Pareto efficiency, walk-away safety, latency and
  cost are reported separately, never mixed in.
- **Faithful LLM agents.** Moves go through function calling constrained to legal
  options; the model's decision stands (only snap-to-grid and the walk-away floor can
  adjust it, and both are logged), unusable replies are re-asked and then counted as
  errors, and private economics never reach the model.
- **Any OpenAI-compatible endpoint.** Real USD cost when the provider reports it
  (OpenRouter does); cost ceilings and time estimates before every run.
- **Self-contained dashboards.** One offline HTML file per run: leaderboards with live
  filters, heatmaps, cost overviews, and a round-by-round replay of every negotiation
  including messages, reasoning and adjusted moves. Duels get a plain-language page.
- **Reproducibility built in.** Decoding settings pinned in configs, and every trace
  records the code version, run time, fingerprints of every prompt and instruction, and
  the provider that served each call. `compare` warns when two runs differ in anything
  but the model.

## Quickstart

No API key needed: the scripted agent runs every strategy deterministically.

```bash
git clone https://github.com/agentic-commerce-lab/rfq-bench.git
cd rfq-bench
uv sync --extra dev                 # needs https://docs.astral.sh/uv/

uv run rfq-bench run --overwrite    # shows the run plan, then asks to confirm
uv run rfq-bench report             # strategy scores, effects vs control, CIs
uv run rfq-bench dashboard          # writes results/dashboard.html
```

## Run with an LLM

```bash
cp .env.example .env                # set OPENAI_BASE_URL, OPENAI_API_KEY, RFQ_BENCH_MODEL
uv run rfq-bench doctor --role seller                     # one test call: can the model use tools?
uv run rfq-bench run --agent llm --strategies control,anchoring --opponents hardliner
```

Any endpoint that speaks the OpenAI chat-completions API works, as long as the model
supports function calling. The run plan shows a cost ceiling and a time estimate before
you confirm.

## Duel two models

```bash
uv run rfq-bench run --config configs/duel-sol-opus-smoke.toml    # 4 episodes, about $1
uv run rfq-bench run --config configs/duel-sol-opus.toml          # 64 episodes
```

Each model sells to the other and buys from the other, on the same scenarios with the
same prompt, so any advantage a scenario gives one role cancels out. Copy a duel config
and change `models` to pit other models against each other.

## Commands

| Command | Does |
|---|---|
| `rfq-bench run` | Run a matrix and append immutable traces (plan, cost and time estimate first) |
| `rfq-bench report` | Score a trace file: strategy effects, separate tracks, fidelity, or the duel readout |
| `rfq-bench dashboard` | Build the self-contained HTML dashboard for a trace file |
| `rfq-bench compare A B` | Compare two runs on matched cells, with a condition check and a comparison page |
| `rfq-bench list` | Show strategies, opponents, personas and scenarios |
| `rfq-bench doctor` | Make one real tool call and print the raw response |
| `rfq-bench config init\|show` | Scaffold or validate a TOML config |

Every command takes `--help`. Reusable run setups live in [`configs/`](./configs).

## Documentation

- [Getting started](docs/getting-started.md): install, first runs, configs, LLM setup,
  custom prompts, troubleshooting
- [Methodology](docs/methodology.md): the score, strategies, how an episode runs, metrics,
  how to read results
- [Agent conditions](docs/conditions.md): `scripted`, `llm`, `a2a`, `duel`, `laya`
- [Reports and dashboards](docs/reports-and-dashboards.md): `report`, `dashboard`, `compare`
- [Reproducibility](docs/reproducibility.md): pinned conditions, provenance, archiving and
  repeating runs
- [Limitations](docs/limitations.md): what the results do and don't support
- [Roadmap](docs/roadmap.md): planned lines of work

## Project layout

```
src/rfq_bench/
  core/        pure scoring kernel: contracts, utilities, ZOPA, q / S_s / Δ (no engine, no network)
  strategies/  concession policies (the treatment)        opponents/  scripted opponents
  agent/       OpenAI-compatible client, LLM and Laya negotiators, prompts, cost/time estimates
  runners/     NegMAS bindings and the episode runner
  report/      aggregation, fidelity, compare, duel, dashboards
configs/       ready-made run configs            data/scenarios*/  scenario sets (immutable)
data/runs/     local run archive (git-ignored)   prompts/          markdown prompts, strategies, personas
```

## Status

rfq-bench is research software (v0.1). The offline benchmark, A2A self-play, model duels
and run comparison are implemented and tested; the Laya agent is experimental and needs
a separate Laya server. Scenario sets are still small (5 anchor and 8 price scenarios), so
confidence intervals are wide. See [Limitations](docs/limitations.md) and the
[Roadmap](docs/roadmap.md).

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) first: the
scientific contract in [`AGENTS.md`](./AGENTS.md) (frozen conditions, immutable traces,
pure scoring, no leakage) applies to every change.

## Citing

```bibtex
@software{rfq_bench,
  title  = {rfq-bench: A reproducible benchmark for negotiating AI agents},
  author = {{Shopware AG}},
  year   = {2026},
  url    = {https://github.com/agentic-commerce-lab/rfq-bench}
}
```

## License

[MIT](./LICENSE) © 2026 Shopware AG. Third-party code: see [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).
Negotiations run on [NegMAS](https://negmas.readthedocs.io).
