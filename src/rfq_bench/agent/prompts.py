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

from rfq_bench.agent.framing import priorities_view, walk_away_view
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
    "negotiation. You negotiate over a fixed set of issues, each with a closed list "
    "of allowed options. You know your role, your priorities, and your own walk-away "
    "(bottom line); you do NOT know the other party's costs, priorities, or limits.\n\n"
    "Each turn, call the `submit_move` function exactly once, and ALWAYS set a value\n"
    "for every issue field (they form your package):\n"
    '- action "offer": propose that package, using only each issue\'s allowed options.\n'
    '- action "accept": accept the opponent\'s current standing offer.\n'
    '- action "terminate": walk away because no acceptable agreement is reachable.\n'
    "Do not agree to any deal that is worse for you than your stated walk-away.\n"
    "Follow the negotiation approach provided in the payload."
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


def build_move_tool(issues: list[Issue], *, strict: bool = False) -> dict[str, Any]:
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
    """
    properties: dict[str, Any] = {
        "action": {
            "type": "string",
            "enum": ["offer", "accept", "terminate"],
            "description": "offer a package, accept the standing offer, or walk away",
        },
        "rationale": {"type": "string", "description": "one short sentence"},
    }
    for i in issues:
        options = ", ".join(str(v) for v in i.values)
        properties[i.name] = {
            "type": _enum_type(list(i.values)),
            "description": f"your {i.name} for this package; choose one of: {options}",
        }
    params: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": ["action", "rationale", *[i.name for i in issues]],
    }
    function: dict[str, Any] = {
        "name": "submit_move",
        "description": (
            "Submit your negotiation move. Fill every issue field with one of its allowed "
            "options — your proposed package for an 'offer' (ignored for accept/terminate)."
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
    "control": "Negotiate neutrally toward a fair target. No special anchoring or timing.",
    "boulware": "Hold firm near your best target for most of the session; concede sharply only "
    "as the deadline approaches.",
    "linear": "Concede at a steady, roughly constant rate from your opening toward your "
    "reservation value.",
    "conceder": "Concede quickly and early, moving toward your reservation value fast.",
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


def _public_issues(issues: list[Issue]) -> list[dict[str, Any]]:
    return [{"name": i.name, "unit": i.unit, "allowed_options": list(i.values)} for i in issues]


def build_user_payload(
    state: NegotiationState,
    approach_instruction: str,
    standing_offer: dict[str, Any] | None,
) -> str:
    """Serialize the business-terms negotiation state the model may see.

    Only the target party's own business view is included — priorities and a
    walk-away, never the raw utilities/weights, and never the opponent's values.
    ``approach_instruction`` is the one per-arm behavioral steer, already resolved
    to text by the caller — a seller strategy (:func:`strategy_instruction`) or a
    buyer persona (:func:`rfq_bench.agent.personas.persona_instruction`). It is
    disposition, never private economics, so it does not widen the leakage surface.
    """
    payload = {
        "your_role": state.role,
        "issues": _public_issues(state.issues),
        "your_priorities": priorities_view(state.prefs, state.issues),
        "your_walk_away": walk_away_view(state.prefs, state.issues),
        "round": state.round,
        "deadline_rounds": state.deadline,
        "standing_offer_from_opponent": standing_offer,
        "opponent_offer_history": state.opponent_offers,
        "your_offer_history": state.my_offers,
        "approach_instruction": approach_instruction,
    }
    return json.dumps(payload, default=str)
