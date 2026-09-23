# Custom system prompts

The LLM agents' **system prompt** is a markdown file. The packaged default lives
inside the package (`src/rfq_bench/agent/prompt_templates/system.md`); you override
it here or via env, without touching code.

Resolution order for each side (buyer / seller), highest precedence first:

1. **Role-specific env path** — `RFQ_BENCH_BUYER_SYSTEM_PROMPT` /
   `RFQ_BENCH_SELLER_SYSTEM_PROMPT` (must exist).
2. **General env path** — `RFQ_BENCH_SYSTEM_PROMPT`, applied to both sides.
3. **`prompts/system_<role>.md`** in this folder — auto-discovered per-role override
   (`system_buyer.md`, `system_seller.md`).
4. **`prompts/system.md`** in this folder — auto-discovered shared override.
5. The packaged default.

So the quickest customization is to copy `system.example.md` to `system.md` (shared)
or `system_buyer.md` / `system_seller.md` (per role) and edit it — no env needed.

Notes:
- The prompt is frozen *within a run* (the same resolved text for every arm). An
  override is a new, explicitly versioned benchmark condition — don't compare its
  results against runs made with a different prompt.
- The negotiation state (role, issues with options best-first, walk-away, the
  strategy/persona instruction, `history`, `rounds_left`, the standing offer) is
  appended automatically as the user message. Keep the system prompt about *how to
  behave and how to respond*. It must still tell the model to call `submit_move` with
  a value for every issue, and explain how to read `your_walk_away` and `history`,
  because the payload carries bare terms without explanatory sentences. Leave the
  per-turn facts to the harness.
- Keep buyer and seller prompts symmetric. In A2A a prompt that differs by role
  confounds the comparison between the two sides.
- `system.example.md` and this README are **not** auto-loaded (only `system.md` /
  `system_<role>.md` are), so they are safe to keep here.

## Custom strategies and personas

Strategies (the seller-side approach) and personas (the buyer-side disposition) are
each one short **guidance instruction** injected into the prompt. Define or override
them with markdown files — one file per name, the **filename stem is the name**:

- `prompts/strategies/<name>.md` — e.g. `prompts/strategies/aggressive.md` adds a
  strategy `aggressive`; `prompts/strategies/boulware.md` overrides the built-in.
- `prompts/personas/<name>.md` — e.g. `prompts/personas/deadline_panic.md` adds a
  persona `deadline_panic`; `prompts/personas/hardball.md` overrides the built-in.

The file's whole text is the guidance. `rfq-bench list` shows the effective set
(new names marked `*`). Use a new name in a run by naming it explicitly, e.g.
`--strategies control,aggressive` or `--personas neutral,deadline_panic` (built-ins
remain the defaults when you pass none).

Caveats:

- A markdown strategy defines only the **LLM guidance** — it works with `--agent llm`
  and `--agent a2a`. Running a *new* strategy in scripted offline mode also needs a
  deterministic policy in `strategies/registry.py`; the built-in seven have both.
- A persona must stay **disposition only** — no private economics (utilities, weights,
  BATNA). The leakage test checks the built-ins **and** every `*.md` file here, so a
  file that uses those words fails the test suite. Use the payload's own terms
  ("walk-away") instead.
- In A2A the sides can exchange short public messages, so guidance about sharing or
  withholding information has an effect there. With `--agent llm` there is no channel.

