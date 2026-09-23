"""FROZEN role prompt and the business-terms state payload sent to the model.

The model sees only what a real negotiator would walk in with: its role, the
product, each issue's concrete options and units, which options it prefers and
how much each issue matters, and its own walk-away (bottom line). It never sees
the calculated 0..1 utilities or issue weights, nor anything about the opponent's
private values — the leakage test enforces this. The role prompt and action
schema are identical across arms; only the one behavioral *approach* instruction
varies (a seller strategy, or a buyer persona in A2A self-play).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from rfq_bench.agent.framing import issues_view, walk_away_terms
from rfq_bench.core.contracts import Issue, Role
from rfq_bench.strategies.base import NegotiationState

# --- System prompt: a markdown file, overridable per run ----------------------
#
# The default lives in ``prompt_templates/system.md`` (shipped with the package)
# and is the single source of truth. It can be overridden without touching code:
#   1. a role-specific or general file path (from env, see agent.settings), or
#   2. a ``prompts/system_<role>.md`` / ``prompts/system.md`` in the working dir.
# The prompt is frozen *within a run* (the same resolved text for every arm); an
# override is a new, explicitly versioned benchmark condition.

# Hardcoded fallback, used only if the packaged markdown cannot be read (e.g. a
# stripped install). Kept byte-identical to prompt_templates/system.md.
_BUILTIN_SYSTEM_PROMPT = (
    "You are a professional procurement/sales negotiator in a bilateral e-commerce "
    "negotiation; your role is in the payload. Each issue has a closed list of options. "
    "You know your own priorities and walk-away, not the other party's costs, "
    "priorities, or limits.\n\n"
    "Your walk-away is the worst deal you may accept. For one issue it is a limit: "
    "`at_most` means never agree to more, `at_least` never to less. For several issues it "
    "is a roughly break-even package (accept only clearly better ones). Never accept "
    "anything worse.\n\n"
    "Each turn, call `submit_move` once with an action and a value for every issue field "
    "(your package):\n"
    '- "offer": propose the package.\n'
    '- "accept": accept the opponent\'s standing offer.\n'
    '- "terminate": walk away; no acceptable deal is reachable.\n'
    "`rationale` is private. `history` holds all moves so far, oldest first, as [who, "
    "offer] plus any message, or a policy note on your own corrected moves. `rounds_left` "
    "includes this turn.\n"
    "Follow the approach_instruction."
)

# Working-directory folder scanned for user overrides (like data/ and results/).
PROMPTS_DIR = "prompts"


def _read_packaged_default() -> str:
    try:
        text = (
            resources.files("rfq_bench.agent")
            .joinpath("prompt_templates/system.md")
            .read_text(encoding="utf-8")
        )
        return text.strip() or _BUILTIN_SYSTEM_PROMPT
    except (FileNotFoundError, OSError, ModuleNotFoundError):
        return _BUILTIN_SYSTEM_PROMPT


DEFAULT_SYSTEM_PROMPT = _read_packaged_default()
# Back-compat: the module-level default prompt.
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def load_system_prompt(path: str | Path) -> str:
    """Read a system-prompt markdown file, stripped. Raises if it is missing/empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"system prompt file not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"system prompt file is empty: {p}")
    return text


def resolve_system_prompt(
    role: Role,
    *,
    explicit: str | Path | None = None,
    general: str | Path | None = None,
    prompts_dir: str | Path = PROMPTS_DIR,
) -> str:
    """Resolve the system prompt for ``role`` (highest precedence first):

    1. ``explicit`` — a role-specific file path (must exist).
    2. ``general`` — a shared file path applied to both roles (must exist).
    3. ``<prompts_dir>/system_<role>.md`` — an auto-discovered per-role override.
    4. ``<prompts_dir>/system.md`` — an auto-discovered shared override.
    5. the packaged default (:data:`DEFAULT_SYSTEM_PROMPT`).

    A path passed explicitly (1, 2) must exist so a typo fails loudly; the
    working-directory files (3, 4) are optional and silently skipped when absent.
    """
    if explicit is not None:
        return load_system_prompt(explicit)
    if general is not None:
        return load_system_prompt(general)
    base = Path(prompts_dir)
    for name in (f"system_{role}.md", "system.md"):
        candidate = base / name
        if candidate.exists():
            return load_system_prompt(candidate)
    return DEFAULT_SYSTEM_PROMPT


def _enum_type(values: list[Any]) -> str:
    """JSON-Schema type for a set of enum option values."""
    if all(isinstance(v, bool) for v in values):
        return "boolean"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "integer"
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return "number"
    return "string"


# A2A message channel: the most of a message the other party receives. The full
# text stays in the trace (flagged when cut), so nothing is lost for audit.
MESSAGE_MAX_CHARS = 300


def build_move_tool(
    issues: list[Issue], *, strict: bool = False, message_channel: bool = False
) -> dict[str, Any]:
    """The function/tool schema for one negotiation move.

    Design for maximum cross-provider fillability: issues are **flat top-level,
    plainly-typed scalar fields** (integer/number/string) with their allowed
    options listed in the *description* — deliberately **not** ``enum``-constrained
    and **not** nested. Some models/providers (Gemini-family via OpenRouter) fill
    plain typed fields fine but return empty arguments for enum-constrained ones,
    so we rely on the negotiator's snap-to-grid to enforce legality instead of the
    schema's enum. Only ``action`` (a small string set) stays an enum.

    ``strict`` (off by default) adds ``function.strict = true`` and
    ``additionalProperties: false`` for providers that support and benefit from it.

    ``message_channel`` (A2A only) adds a public ``message`` field the other party
    reads with the move. It is always present, so strict schemas stay valid; an
    empty string means "no message". ``rationale`` stays private either way.
    """
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": ["offer", "accept", "terminate"],
            "description": "offer the package, accept the standing offer, or walk away",
        },
        "rationale": {
            "type": "string",
            "description": "one short private sentence; never shown to the other party",
        },
    }
    if message_channel:
        properties["message"] = {
            "type": "string",
            "description": (
                f"optional message the other party reads with your move (max "
                f"{MESSAGE_MAX_CHARS} characters); empty string for none. Only the action "
                "and issue fields bind, so any terms you mention must match them."
            ),
        }
    for i in issues:
        options = ", ".join(str(v) for v in i.values)
        properties[i.name] = {
            "type": _enum_type(list(i.values)),
            "description": f"choose one of: {options}",
        }
    fixed = ["action", "rationale", *(["message"] if message_channel else [])]
    params: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": [*fixed, *[i.name for i in issues]],
    }
    function: dict[str, Any] = {
        "name": "submit_move",
        "description": (
            "Submit your move. Every issue field is required: together they are the "
            "package you offer (ignored for accept/terminate)."
        ),
        "parameters": params,
    }
    if strict:
        params["additionalProperties"] = False
        function["strict"] = True
    return {"type": "function", "function": function}


# --- Strategy guidance (the seller-side approach; overridable by md files) ----
#
# Built-in guidance for the seven literature-backed strategies. New strategies (or
# overrides of these) can be defined by dropping `prompts/strategies/<name>.md`
# files in the working dir — see :func:`load_strategy_guidance`. NB: an md file
# defines only the LLM *guidance* (the `--agent llm` / `a2a` steer); running a
# brand-new strategy in scripted offline mode also needs a policy in
# `strategies/registry.py`.

STRATEGY_GUIDANCE: dict[str, str] = {
    "control": "Negotiate neutrally toward a fair target.",
    "boulware": "Hold firm near your best target for most of the session; concede sharply only "
    "as the deadline approaches.",
    "linear": "Concede at a steady, roughly constant rate from your opening toward your walk-away.",
    "conceder": "Concede quickly and early, moving toward your walk-away fast.",
    "tit_for_tat": "Open cooperatively, then match the opponent: concede when they concede, hold "
    "when they hold.",
    "anchoring": "Open with an ambitious but feasible first offer to anchor high, then keep your "
    "later concessions steady and modest.",
    "logrolling": "Trade across issues: concede on issues you value least in exchange for the "
    "issues you value most, proposing packages that could benefit both sides.",
}

# Working-directory folder scanned for strategy-guidance override/definition files.
STRATEGY_DIR = "prompts/strategies"


def read_guidance_dir(path: str | Path) -> dict[str, str]:
    """Read ``<path>/*.md`` into a ``{name: guidance}`` map (name = file stem).

    Missing dir -> empty map. Empty files are skipped. Shared by the strategy and
    persona loaders so both are defined/overridden the same way.
    """
    out: dict[str, str] = {}
    d = Path(path)
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.md")):
        text = f.read_text(encoding="utf-8").strip()
        if text:
            out[f.stem] = text
    return out


def load_strategy_guidance(extra_dir: str | Path = STRATEGY_DIR) -> dict[str, str]:
    """Built-in strategy guidance, with any ``prompts/strategies/*.md`` merged on top.

    A file whose stem matches a built-in overrides it; a new stem adds a strategy.
    """
    merged = dict(STRATEGY_GUIDANCE)
    merged.update(read_guidance_dir(extra_dir))
    return merged


def strategy_instruction(strategy: str, guidance: dict[str, str] | None = None) -> str:
    g = guidance if guidance is not None else STRATEGY_GUIDANCE
    return g.get(strategy) or g.get("control") or STRATEGY_GUIDANCE["control"]


def _terms(outcome: dict[str, Any] | None) -> str:
    return ", ".join(f"{k}={v}" for k, v in (outcome or {}).items())


def _beyond_limit(walk: dict[str, Any]) -> str:
    """How a rejected move broke the walk-away, in the same direction the payload uses."""
    for bound in walk.values():
        if isinstance(bound, dict) and "at_most" in bound:
            return f"above your maximum of {bound['at_most']}"
        if isinstance(bound, dict) and "at_least" in bound:
            return f"below your minimum of {bound['at_least']}"
    if "break_even_package" in walk:
        return "worse for you than your break-even package"
    return "worse than your walk-away"


def _policy_note(policy: dict[str, Any], played: dict[str, Any], walk: dict[str, Any]) -> str:
    """Tell the author of a corrected move what it sent and what was played instead.

    Only ever shown to the author, who already knows its own walk-away, so it leaks
    nothing; the opponent sees just the corrected offer. For one issue the note
    names the direction ("above your maximum of 160"), matching ``your_walk_away``.
    """
    sent, now = _terms(policy.get("sent")), _terms(played)
    kind = policy.get("kind")
    if kind == "floor_accept":
        note = f"policy: you tried to accept {sent}, {_beyond_limit(walk)}; countered {now}"
    elif kind == "floor_offer":
        note = f"policy: you offered {sent}, {_beyond_limit(walk)}; played {now}"
    elif kind == "snap":
        note = f"policy: {sent} is not an allowed option; played the nearest, {now}"
    else:
        note = f"policy: your move was corrected; played {now}"
    if policy.get("withheld"):
        note += "; your message was not delivered"
    return note + "."


def _history_entry(
    move: dict[str, Any], role: Role, message_channel: bool, walk: dict[str, Any]
) -> list[Any]:
    """One transcript line from the viewer's side: ``[who, offer]`` or
    ``[who, offer, text]``, where who is "you" or "them".

    ``text`` is the move's message, or — on the viewer's *own* corrected move — a
    policy note saying what it sent and what was played instead. The opponent never
    sees that note (it would reveal the author's walk-away).

    A compact array rather than a keyed object: history is the part of the prompt
    that grows every turn, so per-entry overhead dominates late rounds. Order gives
    the sequence and ``rounds_left`` the timing, so no per-entry round number.
    """
    mine = move["by"] == role
    entry: list[Any] = ["you" if mine else "them", move["offer"]]
    policy = move.get("policy")
    message = move.get("message")
    if mine and policy:
        entry.append(_policy_note(policy, move["offer"], walk))
    elif message_channel and message:
        entry.append(message[:MESSAGE_MAX_CHARS])
    return entry


def build_user_payload(
    state: NegotiationState,
    approach_instruction: str,
    standing_offer: dict[str, Any] | None,
    *,
    message_channel: bool = False,
) -> str:
    """Serialize the business-terms negotiation state the model may see.

    Only the target party's own business view is included — options in its
    preference order and its walk-away, never the raw utilities/weights, and never
    the opponent's values. ``approach_instruction`` is the one per-arm behavioral
    steer, already resolved to text by the caller — a seller strategy
    (:func:`strategy_instruction`) or a buyer persona
    (:func:`rfq_bench.agent.personas.persona_instruction`). It is disposition,
    never private economics, so it does not widen the leakage surface.

    Field order is chosen for provider prompt caching, which reuses an identical
    prompt *prefix*: what is fixed for the whole episode comes first, then the
    arm's instruction, then the append-only history, and only the two fields that
    change every turn come last. With ``message_channel`` (A2A), history entries
    carry the other side's messages — what it chose to say, never anything the
    harness adds.
    """
    walk = walk_away_terms(state.prefs, state.issues)
    payload = {
        # Fixed for the episode (and shared across arms of the same scenario/role).
        "your_role": state.role,
        "issues": issues_view(state.prefs, state.issues),
        "your_walk_away": walk,
        # Fixed for this arm.
        "approach_instruction": approach_instruction,
        # Grows by appending only, so each turn's prompt extends the last one.
        "history": [_history_entry(m, state.role, message_channel, walk) for m in state.history],
        # Changes every turn: kept last so it never breaks the cached prefix.
        "rounds_left": max(0, state.deadline - state.round),
        "standing_offer_from_opponent": standing_offer,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
