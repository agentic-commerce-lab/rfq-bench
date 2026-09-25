"""Environment-driven settings for the OpenAI-compatible endpoint.

The base URL, model, and decoding parameters are read from the environment so
any OpenAI-compatible provider can back the fixed agent without code changes.
The resolved values are snapshotted into every trace (see ``to_llm_config``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rfq_bench.agent.prompts import resolve_system_prompt
from rfq_bench.core.contracts import LLMConfig, Role


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini", alias="RFQ_BENCH_MODEL")
    temperature: float = Field(default=0.0, alias="RFQ_BENCH_TEMPERATURE")
    # Generous by default: reasoning models spend output budget "thinking" and a
    # too-low cap truncates the function call, which shows up as a hard error.
    max_tokens: int = Field(default=4096, alias="RFQ_BENCH_MAX_TOKENS")
    # Bounded re-asks when a reply has no usable move (e.g. a reasoning model that
    # narrates its move instead of calling the tool). Each re-ask is a real billed
    # call with a stricter corrective nudge; the count is recorded per turn. 0
    # disables. Faithful — it re-elicits the model's own move, never fabricates one.
    max_reasks: int = Field(default=2, alias="RFQ_BENCH_MAX_REASKS")
    # Retries of the *identical* request when the provider fails transiently: a
    # 200 with no choices (OpenRouter's upstream-error envelope), HTTP 429/5xx, or
    # a connection/timeout error. Exponential backoff: base, 2x, 4x, ... seconds.
    # Separate from re-asks — the model did nothing wrong, so no corrective nudge
    # and no effect on reask_count. A content-filter block is never retried.
    transient_retries: int = Field(default=3, alias="RFQ_BENCH_TRANSIENT_RETRIES")
    transient_backoff_s: float = Field(default=1.0, alias="RFQ_BENCH_TRANSIENT_BACKOFF")
    seed: int | None = Field(default=7, alias="RFQ_BENCH_SEED")
    timeout: float = Field(default=60.0, alias="RFQ_BENCH_TIMEOUT")
    # Ask the endpoint to report real spend in the response usage (OpenRouter's
    # `usage.cost`, in USD). Harmless on providers that ignore it; set false if an
    # endpoint rejects the extra request field.
    include_cost: bool = Field(default=True, alias="RFQ_BENCH_INCLUDE_COST")
    # Strict function calling (adds function.strict + additionalProperties:false).
    # OFF by default: some providers return empty tool arguments under strict, and
    # legality is already guaranteed by the negotiator's snap-to-grid. Turn on for
    # providers that support strict and benefit from it.
    strict_tool: bool = Field(default=False, alias="RFQ_BENCH_STRICT_TOOL")
    # How the move tool is requested. "forced" names the function in tool_choice
    # (the default); "required" asks for any tool call; "auto" lets the model
    # decide. Some models reject forced/required tool_choice (e.g. Claude with
    # extended thinking: 'tool_choice: type "tool" and "any" are not supported'),
    # so use "auto" there — a reply without a tool call is re-asked as usual.
    # Recorded in the trace's LLMConfig: changing it is a new condition.
    tool_choice: Literal["forced", "required", "auto"] = Field(
        default="forced", alias="RFQ_BENCH_TOOL_CHOICE"
    )
    # System-prompt overrides (markdown files). A general file applies to both
    # sides; the per-role files win for that role. Unset -> auto-discover
    # ./prompts/system[_<role>].md, else the packaged default. See agent.prompts.
    system_prompt_file: str | None = Field(default=None, alias="RFQ_BENCH_SYSTEM_PROMPT")
    buyer_system_prompt_file: str | None = Field(
        default=None, alias="RFQ_BENCH_BUYER_SYSTEM_PROMPT"
    )
    seller_system_prompt_file: str | None = Field(
        default=None, alias="RFQ_BENCH_SELLER_SYSTEM_PROMPT"
    )

    def system_prompt_for(self, role: Role) -> str:
        """Resolve the system prompt for ``role`` from env overrides / files / default."""
        role_file = (
            self.buyer_system_prompt_file if role == "buyer" else self.seller_system_prompt_file
        )
        return resolve_system_prompt(role, explicit=role_file, general=self.system_prompt_file)

    def to_llm_config(self) -> LLMConfig:
        return LLMConfig(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            tool_choice=self.tool_choice,
        )
