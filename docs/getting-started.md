# Getting started

This page takes you from a fresh clone to your first scripted run, your first LLM run,
and your first model duel.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) for the environment and dependencies
- For LLM runs: an OpenAI-compatible endpoint and API key. [OpenRouter](https://openrouter.ai)
  works well: one key reaches most model families, and it reports real USD cost per call.
  The tested models must support function calling.

## Install

```bash
git clone https://github.com/agentic-commerce-lab/rfq-bench.git
cd rfq-bench
uv sync --extra dev
uv run pytest -m "not llm"     # optional: the offline test suite, no network needed
```

## 1. A first run without an API key

The scripted agent plays every strategy deterministically against the scripted
opponents. It costs nothing and finishes in seconds.

```bash
uv run rfq-bench list                  # strategies, opponents, personas, scenarios
uv run rfq-bench run --overwrite       # prints the run plan, then asks to confirm
uv run rfq-bench report                # the strategy-effects table
uv run rfq-bench dashboard             # results/dashboard.html, open it in a browser
```

`run` always shows the plan first (matrix, episode count, output file, and for LLM runs
a cost ceiling and a time estimate). Answer `y`, or pass `-y` / `--yes` in scripts.
Traces are appended to `results/offline_v0.jsonl`; `--overwrite` starts the file fresh.

## 2. Reusable configs

Everything you can pass as a flag can live in a TOML file. The repository ships ready
configs in `configs/`:

| Config | What it runs |
|---|---|
| `offline-full.toml` | The full scripted matrix |
| `seller-anchoring-llm.toml` | One LLM strategy arm as seller |
| `a2a-smoke.toml`, `a2a-price-smoke.toml` | Small LLM-vs-LLM self-play checks |
| `a2a-grid.toml`, `a2a-price.toml` | Full self-play grids (multi-issue / price only) |
| `duel-*-opus-smoke.toml`, `duel-*-opus.toml` | Model duels (4-episode smoke test and full run) |
| `laya-smoke.toml`, `laya-full.toml` | The experimental Laya agent |

```bash
uv run rfq-bench config init                 # scaffold a commented ./rfq-bench.toml
uv run rfq-bench config show --config configs/duel-sol-opus.toml
uv run rfq-bench run --config configs/offline-full.toml
```

Precedence is **explicit flag > config file > built-in default**, so a flag you type
still wins. An `rfq-bench.toml` in the working directory is picked up automatically.
Unknown keys are rejected, so typos fail loudly.

```toml
[run]
agent       = "llm"
scenarios   = ["single_price_a", "multi_pdw_a"]
strategies  = ["control", "boulware", "anchoring"]
opponents   = ["hardliner", "reciprocal"]
concurrency = 8
out         = "results/offline_llm.jsonl"

[llm]                      # the decoding condition; overrides .env for this run
temperature = 0.0
max_tokens  = 4096
seed        = 7
tool_choice = "auto"

[dashboard]
control = "control"
n_boot  = 2000
```

`report` and `dashboard` accept the same `--config` to find the traces and scenario
data a run produced.

## 3. Connecting an LLM

Copy `.env.example` to `.env` and set at least:

```bash
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-...
RFQ_BENCH_MODEL=openai/gpt-6-sol
```

The endpoint, key and model come from the environment. Everything that defines the
experimental condition (temperature, max tokens, seed, `tool_choice`, re-asks) should be
pinned in the config's `[llm]` table instead; the run plan marks any value still taken
from `.env`. `.env.example` documents every variable.

Check that the model can make a tool call before spending money on a run:

```bash
uv run rfq-bench doctor --role seller    # one real call; prints the raw response
```

Then run a small LLM matrix:

```bash
uv run rfq-bench run --agent llm --strategies control,anchoring --opponents hardliner
```

### What the model sees and may do

The model sees the negotiation in business terms: its role, the product, each issue's
options and units, its priorities and its walk-away. It never sees raw utilities or the
opponent's economics (a leakage test enforces this). Every move is submitted through a
function call whose arguments are limited to legal options. Its decision stands; only
two shop rules can change it, and both are logged:

- **Snap to grid.** An off-grid value moves to the nearest legal option.
- **Reservation floor.** A quote or accept below the agent's own walk-away is raised to it.

A reply without a usable move is re-asked up to `max_reasks` times with a stricter
instruction. If it still fails, the episode is recorded as an error and excluded from
the score (the error rate is reported separately). No scripted value is ever
substituted for the model's move. Transient provider failures (HTTP 429/5xx, dropped
connections, empty completions) are retried with exponential backoff, and a failed
episode is re-run up to `--retries` times.

### Cost and time before you start

For every LLM run the plan prints a cost ceiling (every episode running to its deadline)
and a wall-clock estimate. Both use measured per-call cost and latency from your past
traces in `results/` when they exist, and published OpenRouter pricing otherwise. On
OpenRouter the plan also shows your remaining credit and warns when the ceiling exceeds
it. Agreements usually end episodes early, so actual spend is lower.

## 4. A first model duel

A duel lets two models negotiate against each other, each as buyer and as seller. Run
the 4-episode smoke test first; it costs about a dollar:

```bash
uv run rfq-bench run --config configs/duel-sol-opus-smoke.toml
uv run rfq-bench report --config configs/duel-sol-opus-smoke.toml
```

Check that the error rate is 0% and the cost per episode looks sane, then run the full
config (64 episodes). The run writes a duel dashboard next to the traces. See
[Agent conditions](conditions.md#model-duel---agent-duel) for how the duel is scored.

To duel other models, copy a duel config and change `[run].models`, `out` and
`dashboard_out`. Use pinned model ids rather than `~latest` aliases.

## Custom prompts, strategies and personas

All model-visible guidance is markdown and can be changed without touching code:

- **System prompt.** Copy `prompts/system.example.md` to `prompts/system.md` (both
  roles) or `prompts/system_buyer.md` / `prompts/system_seller.md`. Environment
  variables `RFQ_BENCH_SYSTEM_PROMPT`, `RFQ_BENCH_BUYER_SYSTEM_PROMPT` and
  `RFQ_BENCH_SELLER_SYSTEM_PROMPT` point at other files. `rfq-bench doctor` shows which
  prompt resolved.
- **Strategies and personas.** Drop `prompts/strategies/<name>.md` or
  `prompts/personas/<name>.md`; the file name is the arm name and the text is the
  guidance the model receives. `rfq-bench list` marks added names with `*`.

A changed prompt is a new condition. Every trace records a fingerprint of each prompt
and instruction, and `compare` warns when they differ between runs. Keep custom personas
disposition-only (no private economics). Details in [`prompts/README.md`](../prompts/README.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `tool_choice: type "tool" and "any" are not supported for this model` | The model (e.g. Claude with extended thinking) rejects forced tool calls. Set `tool_choice = "auto"` in `[llm]`, and use it for every model you compare. |
| Many errors with `finish_reason=length` | Reasoning used up the output budget. Raise `max_tokens` in `[llm]`. |
| `provider returned no completion` warnings | Upstream provider failures. They are retried automatically; lower `concurrency` if they persist. |
| `unknown scenario(s)` | The scenario ids belong to another dataset. Pass `--data data/scenarios_price` (or set `[run].data`). |
| Cost estimate "unavailable" | No run history for the model and no published pricing from the endpoint. The run still works. |
