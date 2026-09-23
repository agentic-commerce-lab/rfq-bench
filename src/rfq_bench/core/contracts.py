"""Canonical data model shared by the offline runner and (later) the shop runner.

Utilities are expressed on a normalized [0, 1] scale (weighted-additive over
issues). Private economics — the ``PartyPreferences`` blocks and BATNAs — must
never enter any model-visible payload; see ``agent.prompts`` and the leakage
test.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["buyer", "seller"]
ROLES: tuple[Role, Role] = ("buyer", "seller")

Action = Literal["offer", "accept", "reject", "terminate"]
OutcomeKind = Literal["agreement", "no_deal", "walk_away"]


class ScenarioKind(StrEnum):
    single_issue = "single_issue"
    multi_issue = "multi_issue"
    no_zopa_diagnostic = "no_zopa_diagnostic"


class Issue(BaseModel):
    """One negotiable variable with a closed, ordered set of legal values.

    Values are referenced by position (index) everywhere else, which keeps the
    model JSON-stable and avoids float dict-key ambiguity.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    unit: str | None = None
    values: list[Any] = Field(min_length=1)

    def index_of(self, value: Any) -> int:
        return self.values.index(value)


class IssuePreference(BaseModel):
    """A party's weight and per-value utility for a single issue.

    ``value_utilities`` is aligned to :attr:`Issue.values` by index; each entry
    is in [0, 1].
    """

    model_config = ConfigDict(frozen=True)

    issue: str
    weight: float = Field(gt=0)
    value_utilities: list[float]

    @model_validator(mode="after")
    def _check_range(self) -> IssuePreference:
        for u in self.value_utilities:
            if not (0.0 <= u <= 1.0):
                raise ValueError(f"value utility {u} for issue {self.issue!r} outside [0, 1]")
        return self


class PartyPreferences(BaseModel):
    """PRIVATE economics for one party. Never send this to the model."""

    model_config = ConfigDict(frozen=True)

    role: Role
    issues: list[IssuePreference] = Field(min_length=1)
    batna: float = Field(ge=0.0, le=1.0, description="Reservation utility; disagreement outcome")

    def preference_for(self, issue_name: str) -> IssuePreference:
        for ip in self.issues:
            if ip.issue == issue_name:
                return ip
        raise KeyError(f"no preference for issue {issue_name!r}")


class Scenario(BaseModel):
    """A single negotiation case. Contains private economics for both parties."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: ScenarioKind
    product: str
    outcome_space_version: str
    issues: list[Issue] = Field(min_length=1)
    buyer: PartyPreferences
    seller: PartyPreferences
    deadline_rounds: int = Field(gt=0)
    seed: int

    @model_validator(mode="after")
    def _check_alignment(self) -> Scenario:
        by_name = {i.name: i for i in self.issues}
        if self.buyer.role != "buyer" or self.seller.role != "seller":
            raise ValueError("buyer/seller preference roles are swapped")
        for prefs in (self.buyer, self.seller):
            for ip in prefs.issues:
                issue = by_name.get(ip.issue)
                if issue is None:
                    raise ValueError(f"preference references unknown issue {ip.issue!r}")
                if len(ip.value_utilities) != len(issue.values):
                    raise ValueError(
                        f"{prefs.role} utilities for {ip.issue!r} do not align with legal values"
                    )
        if self.kind == ScenarioKind.single_issue and len(self.issues) != 1:
            raise ValueError("single_issue scenario must have exactly one issue")
        if self.kind == ScenarioKind.multi_issue and len(self.issues) < 2:
            raise ValueError("multi_issue scenario must have at least two issues")
        return self

    def preferences(self, role: Role) -> PartyPreferences:
        return self.buyer if role == "buyer" else self.seller


# --- Trace: the immutable, scorable record of one episode ---------------------


class LLMConfig(BaseModel):
    """Frozen decoding configuration, snapshotted into every trace."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    base_url: str
    model: str
    temperature: float
    max_tokens: int
    seed: int | None = None


class Step(BaseModel):
    """One party action within an episode.

    ``rationale`` is the acting agent's own one-line explanation for the move,
    when it produces language (the LLM agent returns it in its JSON response).
    Deterministic policies leave it ``None``. ``fallback`` records that a safety
    rail overrode the model's chosen action/outcome, so the rationale may
    describe an intent that was not actually played.
    """

    model_config = ConfigDict(frozen=True)

    round: int
    party: Role
    action: Action
    outcome: dict[str, Any] | None = None
    rationale: str | None = None
    # The played ``outcome`` was adjusted from what the model returned by a benign,
    # faithful rule: an off-grid value snapped to the nearest legal tier, or a
    # below-floor quote clamped up to the shop's reservation floor. Never a
    # scripted-strategy substitution. ``intended_*`` record what the model tried.
    adjusted: bool = False
    adjust_reason: str | None = None
    # Which policy constraint fired: "snap" | "floor_offer" | "floor_accept".
    adjust_kind: str | None = None
    intended_action: str | None = None
    intended_outcome: dict[str, Any] | None = None
    # This turn produced an unusable reply and aborted the episode.
    error: bool = False
    error_reason: str | None = None
    # The model's raw reply (verbatim) and its chain-of-thought, when the LLM agent
    # produced them. These are for human audit only — never fed back to any model.
    raw_response: str | None = None
    reasoning: str | None = None
    # Corrective re-asks the LLM agent needed to elicit this move (0 = first try).
    reask_count: int = 0
    # A2A only: the public message sent with this move, in full (the other party
    # received at most MESSAGE_MAX_CHARS of it; ``message_truncated`` says if cut).
    # Unlike ``rationale``, this WAS shown to the opponent.
    message: str | None = None
    message_truncated: bool = False
    # A policy constraint fired on this move, so the message was not delivered.
    message_withheld: bool = False


class Trace(BaseModel):
    """Immutable episode record. Scored independently of NegMAS.

    Every field needed to reproduce and audit the run lives here, so scoring
    never re-invokes the model.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    scenario_id: str
    outcome_space_version: str
    strategy: str
    opponent: str
    target_role: Role
    first_speaker: Role
    seed: int
    llm_config: LLMConfig
    steps: list[Step]
    outcome_kind: OutcomeKind
    agreement: dict[str, Any] | None
    # Realized utilities per party, on the normalized [0, 1] scale.
    utilities: dict[Role, float]
    rounds_to_close: int
    # The effective deadline the episode ran under (the scenario's own deadline,
    # or lower when capped by ``max_rounds``). None on traces written before this
    # field existed; readers then fall back to the scenario's deadline.
    deadline: int | None = None
    latency_s: float
    token_cost: int
    # Real spend in USD, when the provider reports it (e.g. OpenRouter's
    # usage.cost); None when no cost was returned. In offline mode this is the
    # tested agent's spend; in A2A mode it is the two sides' spend combined.
    cost_usd: float | None = None
    # --- A2A self-play condition (offline traces leave these at their defaults) ---
    # "offline" = one LLM/scripted agent vs a scripted opponent (the v0.1 suite).
    # "a2a"     = LLM buyer persona vs LLM seller strategy (self-play).
    mode: Literal["offline", "a2a"] = "offline"
    # The buyer persona in A2A mode; None in offline mode. ``strategy`` carries the
    # seller strategy in A2A mode, so (persona, strategy) names the cell.
    persona: str | None = None
    # Per-role token and USD spend, so both sides of an A2A episode are visible in
    # cost reporting. Offline traces record only the tested agent's role.
    token_cost_by_role: dict[Role, int] = Field(default_factory=dict)
    cost_usd_by_role: dict[Role, float] = Field(default_factory=dict)
    # Prompt tokens sent per LLM side, and how many the provider served from its
    # prompt cache — the cache hit rate is cached / prompt. Empty when unreported.
    prompt_tokens_by_role: dict[Role, int] = Field(default_factory=dict)
    cached_tokens_by_role: dict[Role, int] = Field(default_factory=dict)
    # The tested agent produced an unusable reply (see Step.error); the episode is
    # aborted and EXCLUDED from scoring. The error rate is reported separately.
    errored: bool = False
    error_reason: str | None = None
    # Separate-track validity/safety flags (never folded into the score).
    validity: dict[str, bool] = Field(default_factory=dict)
