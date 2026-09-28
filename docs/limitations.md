# Limitations

Read this before drawing conclusions. rfq-bench is strong on **internal validity and
auditability** (frozen conditions, immutable traces, scoring outside the engine) and
deliberately limited in **external validity and statistical power**. Rule of thumb: read
the difference with its CI, not the score alone; check the exclusions; treat A2A and duel
numbers as results on one fixed setup.

## By design

- **One controlled comparison at a time.** The core benchmark varies only the strategy;
  a duel varies only the model. Neither is a general leaderboard, and neither says how a
  model would do with other prompts, tools or opponents.
- **The score measures one thing: value captured above the walk-away.** Fairness, joint
  surplus, efficiency, safety, speed and cost are separate tracks. A high score can mean
  exploiting a weak opponent rather than skill, especially on single-issue scenarios,
  which are zero-sum.
- **A closed, static world.** Bilateral stacked alternating offers, a fixed issue set,
  exact weighted-additive utilities, small outcome spaces. No dynamic issues, outside
  options, multiple parties, or interacting preferences.
- **Synthetic scenarios.** The scenarios are hand-built B2B cases, not real negotiations.

## Statistics

- **Few scenarios.** CIs cluster by scenario, and the anchor set has 5 scenarios (the
  price suite 6 scored, plus 2 no-ZOPA diagnostics). Intervals are coarse; only large
  effects separate cleanly.
- **Equal weighting.** The macro-average weights every cell equally, regardless of `n`.
- **Selection effects.** Degenerate cells, no-ZOPA cells and errored episodes are excluded.
  A model that fails on hard cases gets a smaller, easier `n`. Check the counts.
- **Seeds do not add samples.** Scripted policies are deterministic, and LLM requests use
  one fixed seed, so extra seeds repeat episodes instead of adding independent ones.

## LLM agents

- **Two shop rules can change a move** (snap-to-grid, reservation floor). Both are logged
  and rare; nothing scripted is ever substituted.
- **Nondeterminism.** Hosted models are not fully deterministic at temperature 0, and a
  provider can update a model behind the same id. See [Reproducibility](reproducibility.md).
- **Tool-calling dependence.** The tested model must support function calling. Some
  models reject forced tool calls (use `tool_choice = "auto"`), and reasoning models can
  run out of output tokens (raise `max_tokens`). Unusable replies are re-asked, then
  counted as errors.
- **Cost is only exact when the provider reports it** (OpenRouter's `usage.cost`).
- **`--max-rounds` changes outcomes.** Capped runs are not comparable with full-deadline
  runs.
- **Fingerprints, not full text.** Traces store sha256 fingerprints of prompts and
  instructions, which detect a change but do not contain the text. Keep the prompt files
  under version control.

## A2A self-play and duels

- **Strategies and personas are one-line prompt steers**, not provably distinct policies.
  The fidelity check measures whether a strategy was played; nothing checks personas.
- **Single-factor machinery.** The persona sits on the opponent axis, so there is no
  persona × strategy interaction term.
- **Arbitrary baselines.** Seller Δ is measured against `control`, buyer Δ against
  `neutral`: conventions, not validated zeros.
- **Self-play uses the same model on both sides.** Behaviour is correlated, and a
  stronger buyer cancels a stronger seller. Use a duel to compare models.
- **A duel compares two models.** Rankings across more models built from separate duels
  assume the gaps add up, which nothing has tested yet (see the [roadmap](roadmap.md)).
- **About twice the cost** of a single-LLM run, and fully nondeterministic.
