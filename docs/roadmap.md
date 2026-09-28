# Roadmap

Future lines of work, roughly in order of value. None of this is built yet. Each item
that changes what a run measures will be a new, versioned condition, so existing results
stay comparable.

## Comparing many models

Today a duel takes exactly two models and `compare` exactly two files.

- **Joint rating model.** Fit all matchups at once: the seller's score explained by the
  seller's selling skill minus the buyer's buying skill, plus a scenario effect. That
  gives every model a selling, a buying and an overall rating, each with a
  scenario-clustered CI, instead of stitching pairwise gaps together.
- **Leaderboard command and page.** `rfq-bench leaderboard <duel files…>`: check that all
  files share one condition, fit the ratings, report ties where CIs overlap, and render a
  ranking, a seller × buyer matchup heatmap, skill by role, and cost against score.
- **Tournament runner.** `--agent tournament --models A,B,C,…`: every cross matchup plus
  each model against itself once. Separate duels currently repeat the shared model's
  self-play in every run.
- **Study designs.** Support both a star design (every model against one reference, cost
  linear in the number of models) and round-robin (quadratic, but reveals matchup effects),
  and a hybrid that adds direct duels between close contenders.

## Statistical power and robustness

- **Larger scenario sets.** The 50-scenario validation milestone, then the 500-scenario
  target: generated scenarios with validated hidden economics, ZOPA and Pareto checks.
- **Multi-issue duels.** Duels on price, delivery and warranty, where trading across
  issues matters and joint surplus becomes informative.
- **Measured run-to-run noise.** Repeat runs as a standard step, and report differences
  relative to that noise floor.
- **Provider pinning.** Optionally pin OpenRouter providers (no fallbacks) so a repeat is
  served by the same backend.
- **Model-side seeds.** Forward per-episode seeds to models that support them, so
  repetitions become independent samples.

## Richer negotiations

- **Opponent and persona fidelity.** Extend the strategy-fidelity check to personas, so
  "the persona changed behaviour" becomes measurable.
- **Interaction terms.** Model persona × strategy effects in A2A instead of the
  single-factor Δ.
- **More model families** as tested agents and as opponents.
- **Human baselines.** Human negotiators on the same scenarios, as an external reference.

## Laya

- **Cascade (`--agent cascade`).** Laya decides each turn and escalates only the turns it
  is unsure about to the LLM negotiator, to measure how much cost that saves for how much
  value. See the design in [`laya-agent-concept.md`](laya-agent-concept.md).
- **Alternative move designs** (package choice, gates only) and threshold calibration on
  labelled data.

## Connected mode

- **Shop runner.** Map e-commerce quote events (for example Shopware quotes being created
  or updated) to the same canonical negotiation state, run the same negotiator, and return
  one validated action: first in
  shadow mode against historical quotes, then with human-approved counteroffers. The
  scenario schema, outcome space, scoring and trace format are already shared so this can
  attach without changing the core. Connected-mode scores will carry their own mode label
  and never be pooled with offline results.

## Tooling

- **Screenshot export.** `rfq-bench dashboard --export-png` for cropped report images.
- **Prompt text in traces.** Store the resolved prompt text alongside its fingerprint, so a
  trace is fully self-describing.
