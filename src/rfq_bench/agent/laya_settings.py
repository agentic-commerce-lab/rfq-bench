"""Settings for the Laya decision-model agent (local HTTP server).

Laya is a non-generative decision model served locally; we talk to it over
loopback HTTP (``POST {base_url}{predict_path}`` with ``{"state", "questions"}``).
Everything is env-driven so no endpoint is hardcoded. See docs/laya-agent-concept.md.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rfq_bench.core.contracts import LLMConfig


class LayaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Local Laya server (the skill's FastAPI sidecar defaults to uvicorn's :8000).
    base_url: str = Field(default="http://127.0.0.1:8000", alias="RFQ_BENCH_LAYA_URL")
    predict_path: str = Field(default="/predict", alias="RFQ_BENCH_LAYA_PREDICT_PATH")
    timeout: float = Field(default=60.0, alias="RFQ_BENCH_LAYA_TIMEOUT")
    # Informational: which checkpoint the server loaded (English by default). Recorded
    # into the trace for provenance; the actual model is chosen server-side.
    checkpoint: str = Field(default="english", alias="RFQ_BENCH_LAYA_CHECKPOINT")
    # Decision thresholds (choose on labelled data; see the concept's eval plan).
    # Accept the standing offer only when P(accept) clears this AND it is at/above
    # the reservation floor.
    accept_threshold: float = Field(default=0.6, alias="RFQ_BENCH_LAYA_ACCEPT_THRESHOLD")
    # Walk away when P(a deal is still reachable) falls below this.
    walk_threshold: float = Field(default=0.25, alias="RFQ_BENCH_LAYA_WALK_THRESHOLD")

    def to_llm_config(self) -> LLMConfig:
        """A uniform config snapshot so Laya traces slot into the same Trace shape."""
        return LLMConfig(
            base_url=self.base_url,
            model=f"laya:{self.checkpoint}",
            temperature=0.0,
            max_tokens=0,
            seed=None,
        )
