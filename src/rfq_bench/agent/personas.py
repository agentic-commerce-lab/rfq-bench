"""Buyer personas for A2A self-play.

A **persona** is the buyer-side counterpart to a seller **strategy**: the single
per-arm behavioral steer injected into the frozen role prompt (as the payload's
``approach_instruction``). Personas vary *disposition* only — tone, urgency,
price-sensitivity, risk tolerance, aggressiveness, cooperativeness. They must
NEVER encode private economics: the buyer's utilities, issue weights, BATNA, and
ideal come from the scenario and stay identical across personas, so ``q`` remains
comparable persona-to-persona and the leakage guarantee is unchanged.

For this reason the guidance strings deliberately avoid the words the leakage
test screens for (``utilit``, ``weight``, ``batna``): a persona is how the buyer
*behaves*, not what it privately values.
"""

from __future__ import annotations

from pathlib import Path

from rfq_bench.agent.prompts import read_guidance_dir

# Each persona is one short behavioral instruction, mirroring STRATEGY_GUIDANCE.
PERSONA_GUIDANCE: dict[str, str] = {
    # Neutral baseline: the businesslike buyer with no special disposition.
    "neutral": "Negotiate neutrally and businesslike toward a fair deal. No special "
    "urgency, aggression, or reluctance.",
    # Extremely price-sensitive; grinds for discounts and yields slowly on price.
    "bargain_hunter": "You are intensely focused on getting the lowest possible price. "
    "Push hard for discounts, resist price concessions, and treat a low price as the "
    "main measure of a good deal.",
    # Deadline-driven; values a fast close and concedes to get there.
    "time_pressured": "You are under real time pressure and need to close quickly. "
    "Prefer reaching agreement fast over squeezing out the last bit of value, and "
    "concede reasonably to avoid dragging the negotiation out.",
    # Cooperative integrator; seeks fair, mutually beneficial packages.
    "relationship_builder": "You value a good long-term working relationship. Negotiate "
    "cooperatively, look for mutually beneficial trade-offs across the issues, and aim "
    "for a fair outcome rather than winning at the other side's expense.",
    # Aggressive; anchors extreme, concedes little, ready to threaten walking.
    "hardball": "You negotiate aggressively. Open with an ambitious position, give ground "
    "only grudgingly, apply pressure, and be willing to signal that you will walk away to "
    "extract better terms.",
    # Cautious; prizes certainty and safety over squeezing the best terms.
    "risk_averse": "You are cautious and prize certainty. Favor safe, dependable terms, "
    "avoid brinkmanship, and do not push the negotiation close to breaking down for a "
    "marginally better deal.",
}

PERSONAS: tuple[str, ...] = tuple(PERSONA_GUIDANCE)

# Working-directory folder scanned for persona override/definition files. Drop a
# ``prompts/personas/<name>.md`` in here to override a built-in persona or add a
# new one — a persona is pure disposition, so a markdown file fully defines it.
PERSONA_DIR = "prompts/personas"


def load_persona_guidance(extra_dir: str | Path = PERSONA_DIR) -> dict[str, str]:
    """Built-in persona guidance, with any ``prompts/personas/*.md`` merged on top.

    A file whose stem matches a built-in overrides it; a new stem adds a persona.
    """
    merged = dict(PERSONA_GUIDANCE)
    merged.update(read_guidance_dir(extra_dir))
    return merged


def persona_instruction(persona: str, guidance: dict[str, str] | None = None) -> str:
    """Resolve a persona name to its behavioral instruction (falls back to neutral)."""
    g = guidance if guidance is not None else PERSONA_GUIDANCE
    return g.get(persona) or g.get("neutral") or PERSONA_GUIDANCE["neutral"]


def get_persona(persona: str) -> str:
    """Return the guidance for ``persona`` or raise a clear error for an unknown name."""
    try:
        return PERSONA_GUIDANCE[persona]
    except KeyError:
        raise KeyError(f"unknown persona {persona!r}; known: {', '.join(PERSONAS)}") from None
