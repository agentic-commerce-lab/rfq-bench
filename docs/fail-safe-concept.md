# Concept: making episodes fail-safe (fewer silent exclusions)

**Status:** Layers 1–3 implemented (2026-09-23); Layers 0 and 4 still design.
**Author:** l.rump. **Touches:** `agent/llm_client.py`, `agent/negotiator.py`,
`agent/settings.py`, `runners/negmas_engine.py`, `runners/offline.py`,
`core/contracts.py`, `strategies/base.py`, `cli.py`, `report/`.

**Shipped so far:** the **tool call is the sole authoritative move channel** —
`_parse_tool_call` reads the move only from the forced function call and never
recovers one from message content or the reasoning trace (Layer 1, revised: see
§3.1). Recovery is instead a bounded re-ask loop in `LLMNegotiator.act`, gated by
`RFQ_BENCH_MAX_REASKS` (default 2), that drives the model to emit a proper tool
call; the per-turn `reask_count` is recorded on every trace step (Layer 2).
`LLMClient.complete` retries the identical request with exponential backoff on
transient provider failures — a `choices=null` body, HTTP 429/5xx, dropped
connections — logs the provider's error body, and never retries a content-filter
block; a failure that survives those retries is excluded with a
`provider returned no completion` reason and is not re-asked (Layer 3,
`RFQ_BENCH_TRANSIENT_RETRIES` / `RFQ_BENCH_TRANSIENT_BACKOFF`). Layers 0 and 4 are
not yet built.

## 1. Why

Episodes are silently dropped when a model returns no usable move. The canonical case:

```
ERROR — episode excluded: no package in reply (action=None); expected a value per issue
**Negotiating Price Options**
I'm currently processing the user's desire to negotiate price. ...my walk-away threshold is 110...
```

Here a reasoning model emitted its **chain-of-thought as message content** and never called
`submit_move` (or called it with empty arguments). The move information exists — in prose — but
there is no JSON object and no tool call, so `_parse_tool_call` returns `{}`, the negotiator
raises a hard error (`negotiator.py:210`), the engine ends the negotiation
(`negmas_engine.py:156`), and the trace is flagged `error=True` and excluded from scoring.

Two problems:

1. **These losses are recoverable but not recovered.** The model *can* produce a valid move; it
   just didn't format it. We give up after one attempt.
2. **The existing retry doesn't fire.** `cli.py:534 run_with_retries` only re-attempts on a
   raised `Exception` (transport/network). A protocol failure returns a *successful* trace that
   is merely flagged — no exception, so the retry loop never sees it. The `--retries` flag covers
   flaky networks, **not** flaky formatting, which is the dominant loss here.

The hard constraint (`negotiator.py` docstring): **never substitute a scripted move for the
model.** A fail-safe may *re-elicit* the model's own decision, parse it harder, or classify the
failure honestly — it must never invent the negotiation choice. That rules out "fill in a default
package" and rules in "ask again" and "parse better."

## 2. Failure taxonomy

Every unusable reply today collapses into one bucket (`error=True`). Separating them is the
foundation of the whole concept — each class has a different correct response:

| Class | Example | Model's fault? | Correct response |
| --- | --- | --- | --- |
| **Transient infra** | timeout, 429, 5xx, connection reset | no | retry the *call* (backoff); does not count against the model |
| **Truncation** | `finish_reason=length` mid-tool-call | partly (budget) | retry with more headroom; already detected at `negotiator.py:163` |
| **Protocol / format** | reasoning-as-content, empty tool args, prose-wrapped JSON | recoverable | parse harder, then **re-ask**; count retries |
| **Illegal-but-parseable** | off-grid value, below-floor quote | no — a *decision* | already handled (snap / floor); scored as-is |
| **Genuine model failure** | refuses, no move after N re-asks | yes | exclude — but record *why*, per arm |

Only the last row should ever exclude an episode. Today rows 1–3 also exclude it, silently.

## 3. Design: four layers, cheapest first

Each layer catches what the one before it missed, so the expensive re-ask is a last resort.

### Layer 0 — Prevention (prompt + budget)

- **Sharper move contract in the system prompt.** State explicitly: "After any reasoning, you
  **must** call `submit_move` with a value for every issue. Do not put the move in your reply
  text." Cheap, and it attacks the reasoning-as-content case at the source.
- **Reasoning-budget headroom.** Reasoning models spend the output budget thinking; a low
  `max_tokens` truncates the eventual call. Keep the generous default (`settings.py:26`), and
  where the provider supports it, pass a bounded `reasoning_effort` so thinking can't crowd out
  the tool call entirely.

### Layer 1 — Tool calls are the only move channel (revised)

An earlier draft of this layer mined message content and the reasoning trace for a JSON move.
That was rejected: **the tool call is the authoritative channel, nothing else.**

- Message content is the wrong channel; the reasoning trace holds *intermediate, non-final*
  offers (a model reasons "what if I offer 80… no, 120"). Scraping either risks recording a
  move the model did not submit — an integrity violation of "the model's decision stands."
- So `_parse_tool_call` reads the move **only** from the forced function call's arguments. It
  tolerates a code fence around those arguments (still the tool channel) but nothing more. A
  reply with no usable tool call yields `{}` (the message content is kept as `raw` for audit
  only) → a recoverable error that Layer 2 re-asks.

Consequence: a model that *never* emits tool calls is out of scope (as the client already
states) and its episodes are excluded rather than salvaged from prose. That is the honest,
faithful trade — recovery happens by getting a real tool call, not by guessing at one.

### Layer 2 — Bounded re-ask (the core fail-safe)

When Layers 0–1 still yield no move, **re-prompt the same model in the same turn** with a short
corrective message, up to `N` times (default 2):

> "You did not submit a move. Call `submit_move` now with a value for each of: price, delivery,
> warranty. Reply with the tool call only — no explanation."

This is faithful: it is still the model's own decision, merely re-elicited under a stricter nudge
(the same thing a human would do). Requirements:

- **Bounded and recorded.** Cap at `N`; write `reask_count` and the reason onto the trace so the
  effort is auditable. A model that needs 2 re-asks every turn is a finding, not a free pass.
- **Determinism note.** A re-ask is a second stream; like temperature and multi-stream A2A, it
  weakens byte-identical replay (already an offline-only guarantee — see [Reproducibility](reproducibility.md)). Record it so
  replays are explainable.
- **Scope.** Applies to both sides symmetrically (`LLMNegotiator` is shared), so buyer and seller
  get the same treatment and `q` stays comparable.

### Layer 3 — Classify, retry transient, exclude honestly

- **Split the client's error path.** Catch transport errors in `LLMClient.complete` and tag them
  `transient`; let `run_with_retries` (already present) handle those with its backoff. Today a
  timeout that surfaces as a flagged trace instead of an exception escapes that loop.
- **Only genuine failures exclude.** After Layers 1–2 are exhausted, *then* the episode is
  excluded — but tagged with its taxonomy class, not a generic "unusable model reply."

## 4. Integrity guardrails (exclusion must stay unbiased)

Recovering episodes is only half the point; the other half is that **whatever we still exclude
must not bias the comparison.** If one strategy, persona, or model formats worse than another,
dropping its episodes silently skews Δ-vs-control.

- **Exclusion ledger per arm.** Count exclusions by class × strategy × persona × model and print
  it in `rfq-bench report`; surface an exclusion-rate row in the dashboard next to `S_s`/Δ.
- **Threshold warning.** If any arm's exclusion rate exceeds a bound (e.g. 5%), flag the run as
  bias-suspect rather than reporting Δ as if clean.
- **Symmetric, bounded effort.** The re-ask cap is identical across arms, so no arm gets
  unlimited chances to eventually comply — the number of tries is part of what we measure.

## 5. What changes, concretely

| Layer | File | Change |
| --- | --- | --- |
| 0 | `agent/prompt_templates/system.md` | explicit "call the tool, don't narrate the move" clause |
| 1 | `agent/llm_client.py` | tool-call-only move channel; no content/reasoning mining (fence-tolerant args) |
| 2 | `agent/negotiator.py` + `llm_client.py` | re-ask loop on empty move; `reask_count` on the trace |
| 3 | `agent/llm_client.py`, `cli.py` | tag transient vs protocol; route transient into existing `--retries` |
| 4 | `report/aggregate.py`, `report/*` | per-arm exclusion ledger + threshold warning |

New settings (mirroring the `--retries` precedent): `RFQ_BENCH_MAX_REASKS` (default 2) and a
report threshold. All default-on but bounded, so behavior stays reproducible and honest.

## 6. Non-goals

- **No fabricated moves.** The benchmark never fills in a package the model didn't produce.
- **No infinite retries.** Effort is capped and recorded; persistent failure is a scored outcome.
- **No mixing into the immutable offline suite.** As with A2A, any re-ask non-determinism stays
  out of the byte-identical offline guarantee.

## 7. Expected effect

The pasted example (a reasoning model that narrated its move instead of calling the tool) is
exactly what Layer 2's re-ask fixes: a second, stricter prompt almost always yields a proper
tool call — the authoritative channel Layer 1 insists on. Combined with Layer 3 (transient
timeouts no longer counted against the model), the silent-exclusion rate should drop sharply —
and whatever remains is a genuine failure to produce a tool call: visible, classified, and
bias-checked instead of a quiet hole in the results.
