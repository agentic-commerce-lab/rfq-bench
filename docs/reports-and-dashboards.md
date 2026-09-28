# Reports and dashboards

Every number comes from the persisted traces. Nothing here calls a model, so you can
re-score, re-render and compare as often as you like.

| Command | Input | Output |
|---|---|---|
| `rfq-bench report` | one trace file | text tables in the terminal |
| `rfq-bench dashboard` | one trace file | a self-contained HTML page |
| `rfq-bench compare A B` | two trace files | text tables and a comparison page |

`report` and `dashboard` accept `--config <file>` to find the traces and data a config
produced. All three take `--data` when the traces use another scenario set (e.g.
`data/scenarios_price`).

## `report`

```bash
uv run rfq-bench report --traces results/offline_v0.jsonl
```

Prints, depending on the condition:

- the **strategy table**: `S_s`, `Δ` vs control and its CI, `n` per strategy;
- the **separate tracks**: agreement, joint surplus, Pareto share, rounds, `walk✓`,
  revenue, cost per episode;
- for A2A, a second table scoring the **buyer** by persona;
- the **strategy fidelity** check for LLM and Laya agents;
- prompt-cache hit rates and A2A message activity when the traces carry them.

For a duel file it prints the **duel readout** instead: both models' scores and the gap,
skill by role, the gap per scenario, per-matchup metrics, cost by model, serving
providers, and the run's date and code version.

## `dashboard`

```bash
uv run rfq-bench dashboard --traces results/offline_v0.jsonl --out results/dashboard.html
```

One HTML file with everything inlined: open it with a double-click, no server. `run`
refreshes it after every run.

- A leaderboard with a Δ-vs-control forest plot. Filters (scenario, opponent, role, first
  speaker, seed) recompute scores and bootstrap CIs live; per-episode `q` comes from the
  Python kernel, so the numbers always match `report`.
- A cost overview, a strategy × opponent heatmap of mean `q`, and the fidelity panel.
- An episode browser and a **negotiation replay**: offers and utilities per round
  against the ZOPA band, a step scrubber, the action log with messages, and for each LLM
  turn the verbatim reply and (for reasoning models) the chain of thought. Adjusted and
  errored moves are flagged with the reason and what the model tried.

### Duel dashboard

For a duel file, `dashboard` (and the run itself) writes a duel page instead. It reads
top to bottom and always names the models:

1. **Result**: both scores, the winner, the gap in plain words with its likely range,
   and the cost of the test.
2. **Selling vs buying**: the seller's score in each pairing, and each model's lead when
   selling and when buying.
3. **By scenario**: the lead in each negotiation.
4. **Deals & cost**: cost by model (moves, tokens and cost per move, serving providers)
   and per-matchup metrics (deals closed where possible, correct walk-aways, errors,
   moves, time, cost).
5. **Replay**: every negotiation, filterable by matchup, with the same replay as the main
   dashboard.

## `compare`

```bash
uv run rfq-bench compare results/run_a.jsonl results/run_b.jsonl \
    --label-a baseline --label-b candidate
```

Compares two runs of the same condition, typically the same matrix with two models:

- **Matched cells only.** A cell is scenario × strategy × opponent × role × first
  speaker. Only cells scorable in both runs count (seeds averaged within a cell); the
  skipped counts are printed, so extra coverage in one run neither helps nor hurts it.
- **`Δ = S_B − S_A`**, paired per cell, with a scenario-clustered 95% CI, overall and per
  strategy, opponent and scenario.
- **Separate tracks side by side**, on the cells both runs cover.
- **Condition check.** Any difference besides the model is printed as a `WARNING`: mode,
  outcome-space version, per-cell deadline, temperature, max tokens, seed, `tool_choice`,
  and any changed prompt, instruction or tool-schema fingerprint.
- **Comparison page**, `results/compare_<a>_vs_<b>.html` by default (`--html-out`,
  `--no-html`).

This is a between-run readout: it says which run captured more value under this fixed
setup. It is not the within-agent strategy effect and not a general ranking. For two
models, prefer a [duel](conditions.md#model-duel---agent-duel): with self-play both sides
change model at once, and a stronger buyer can cancel a stronger seller.
