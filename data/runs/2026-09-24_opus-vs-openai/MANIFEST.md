# Model duels: Claude Opus 5.5 vs OpenAI, 2026-09-24

Archived copies of the four full duel runs (the source of the LinkedIn write-up). Traces
are immutable: never edit these files; rebuild reports and dashboards from them.

| File | Models | Episodes | Errors | Cost | Config | sha256 |
|---|---|---|---|---|---|---|
| `duel_terra_opus.jsonl` | Claude Opus 5.5 vs GPT-5.6 Terra | 64 | 0 | $5.74 | `configs/duel-terra-opus.toml` | `b2f7830878205a46…` |
| `duel_sol_opus.jsonl` | Claude Opus 5.5 vs GPT-6 Sol | 64 | 0 | $5.68 | `configs/duel-sol-opus.toml` | `db148496a4c8ac9e…` |
| `duel_astra_opus.jsonl` | Claude Opus 5.5 vs GPT-6 Astra | 64 | 0 | $8.20 | `configs/duel-astra-opus.toml` | `f23afc1540bcf832…` |
| `duel_astra_pro_opus.jsonl` | Claude Opus 5.5 vs GPT-6 Astra Pro | 64 | 0 | $17.76 | `configs/duel-astra-pro-opus.toml` | `c7b70aa3f2c1d404…` |

Full sha256:

- `duel_terra_opus.jsonl`: `b2f7830878205a46c2d7cf615d0989c36a7dfc322fa932ce3ed7c827ecd383e6`
- `duel_sol_opus.jsonl`: `db148496a4c8ac9e957b2ddec66251be66825eae00ce75eea8f02b9ee4c4416f`
- `duel_astra_opus.jsonl`: `f23afc1540bcf832891b3c496b47a9767010993ac9bec8b4c276d08db8fd27c4`
- `duel_astra_pro_opus.jsonl`: `c7b70aa3f2c1d404576f7fc1b515b511cf654e511cdcc510ebc2d929147f6db3`

## Condition

- Endpoint: OpenRouter (`https://openrouter.ai/api/v1`); model ids pinned (no `~latest` aliases).
- Decoding (identical in all four files): `{"base_url": "https://openrouter.ai/api/v1", "max_tokens": 4096, "seed": 7, "temperature": 0.0, "tool_choice": "auto"}`.
- Suite: `data/scenarios_price/price_v1.json`, 8 scenarios (6 with a possible deal); strategy
  `control`, persona `neutral`; both opening orders; pairings `full` (both cross matchups plus
  both self-play matchups); one seed; full 16-round deadline.
- Code: produced by the uncommitted working tree on top of `1334745`; the equivalent duel
  code is committed in `d8eaf57`. These traces predate `Trace.provenance` and `Step.provider`
  (added in `aea4789`), so they carry neither; `compare` will say so when you compare a new
  run against them.
- Serving providers were not recorded. An OpenRouter error from that day lists the providers
  it tries for Opus (Azure, Claude Platform on AWS, Amazon Bedrock, Anthropic, Google), so
  calls may have been served by any of them.

## Rebuild from these traces

```bash
R=data/runs/2026-09-24_opus-vs-openai
uv run rfq-bench report    --traces $R/duel_sol_opus.jsonl --data data/scenarios_price
uv run rfq-bench dashboard --traces $R/duel_sol_opus.jsonl --data data/scenarios_price \
  --out results/duel_sol_opus_dashboard.html
```

## Repeat and compare

```bash
uv run rfq-bench run --config configs/duel-sol-opus.toml --out results/duel_sol_opus_rerun.jsonl
uv run rfq-bench compare $R/duel_sol_opus.jsonl results/duel_sol_opus_rerun.jsonl \
  --data data/scenarios_price --label-a 2026-09-24 --label-b rerun
```
