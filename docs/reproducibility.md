# Reproducibility

rfq-bench treats a run as a measurement: it should be repeatable, and when a repeat
differs you should be able to tell why.

## What is frozen

Within a condition these never change between arms: the model version, the system
prompt, the move tool schema, the decoding settings (temperature, max tokens, seed,
`tool_choice`), what private information the agent sees, and the outcome space.
Changing any of them makes a **new condition**: a new mode or output file, documented,
never an edit to an existing one. The scripted offline suite and the anchor scenario
sets are immutable.

## Pin the condition in the config

Put the decoding settings in the config's `[llm]` table, not in `.env`:

```toml
[llm]
temperature = 0.0
max_tokens  = 4096
seed        = 7
tool_choice = "auto"
max_reasks  = 2
strict_tool = false
```

Values set here override the environment for that run. The run plan prints the
effective settings and marks any value still taken from `.env`. Only the endpoint, the
API key and (outside duels) the model belong in the environment. All shipped duel
configs pin `[llm]`.

## What every trace records

- **The full decoding config** (`llm_config`), and in a duel the model for each side
  (`models_by_role`).
- **Provenance** (`provenance`):
  - `code_version`: the git commit, with `+dirty` if there were uncommitted changes;
  - `started_at`: the run start, UTC;
  - sha256 fingerprints of everything the model saw: `system_prompt_<role>`,
    `instruction_<role>.<arm>` (the strategy or persona text) and `tool_schema`.
- **The serving provider** of every LLM move (`steps[].provider`). OpenRouter routes one
  model id to several providers, and they can behave differently.
- Per role: tokens, prompt and cached tokens, and USD cost when reported.

`compare` and the duel report warn when a fingerprint differs between two runs or
changes within one, and when older traces have no provenance to check.

## Keep the traces

`results/` is scratch space and is overwritten by later runs. Copy runs worth keeping,
such as published results or baselines for later comparison, to
`data/runs/<YYYY-MM-DD>_<name>/`: the trace files unchanged plus a `MANIFEST.md` with
checksums, the condition, the config, the code version, and the commands to rebuild and
to repeat. Both folders are git-ignored; share an archived run by attaching it to a
release. See [`data/runs/README.md`](../data/runs/README.md).

## Repeat a run

```bash
uv run rfq-bench run --config configs/duel-sol-opus.toml --out results/duel_sol_opus_rerun.jsonl
uv run rfq-bench compare data/runs/<date>_<name>/duel_sol_opus.jsonl \
    results/duel_sol_opus_rerun.jsonl --data data/scenarios_price \
    --label-a original --label-b rerun
```

Write a repeat to a new file; never append it to an archived one.

## What still varies

- **Temperature 0 is not fully deterministic** for hosted models, and the episode seed is
  not sent to the model (every request uses the `[llm]` seed). Extra `seeds` in an LLM
  run repeat the same requests rather than adding independent samples.
- **Providers can update a model behind the same id.** Pin model ids (no `~latest`
  aliases) and check the recorded providers when a repeat differs.
- **Run-to-run noise.** Repeat an unchanged setup once to see how far results move, and
  treat smaller differences as noise. When several duels share a model, its self-play
  matchup is repeated in each, which gives a free first estimate of that noise.

Scripted runs are fully deterministic and produce byte-identical traces (apart from
latency and start time).
