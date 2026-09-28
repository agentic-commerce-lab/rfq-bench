# Contributing to rfq-bench

Thanks for your interest. rfq-bench is a measurement tool, so the bar for changes is
that results stay comparable and auditable. Please read the scientific contract in
[`AGENTS.md`](./AGENTS.md) §1 before changing anything that touches a run.

## Development setup

```bash
uv sync --extra dev
uv run pytest -m "not llm"                           # the default suite needs no network
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

All four must pass before a pull request is merged; CI runs the same commands.

## Ground rules

- **Frozen dimensions stay frozen.** Model, prompts, tool schema, decoding settings,
  private-information exposure and the outcome space are fixed within a condition.
  Changing one is a new, explicitly versioned condition (new mode, new output file,
  documented), never an edit to an existing one.
- **Traces are immutable.** Never rewrite a trace file; re-scoring reads traces, it
  does not re-run them. Runs archived locally in `data/runs/` are never edited.
- **Scoring is pure.** `src/rfq_bench/core/` must not import from `runners/`,
  `agent/` or anything that calls NegMAS or a network. Keep it deterministic and
  covered by exact-value unit tests.
- **No leakage.** Private economics (values, costs, BATNAs, reservation values, ideal
  points) never enter a model-visible payload. The leakage test guards this; extend it
  when you add a new payload field, strategy or persona.
- **Add, don't replace.** Keep the anchor scenario sets immutable; add scenarios,
  opponents, personas or conditions alongside them.
- **Mark live tests.** A test that calls a real LLM gets `@pytest.mark.llm` so the
  default suite stays offline.

## Pull requests

- One topic per pull request, with tests for new behaviour.
- Update the docs in `docs/` (and the README if a user-facing command changes).
- Use conventional commit prefixes, as in the history: `feat(scope):`, `fix(scope):`,
  `docs:`, `data:`.
- If your change alters what a run measures, say so in the description and name the
  new condition.

## Reporting issues

Open a GitHub issue with the command you ran, the config file, and (for LLM runs) the
model id and endpoint. Never paste API keys; `.env` stays local.
