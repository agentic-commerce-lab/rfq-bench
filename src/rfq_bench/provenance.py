"""Run provenance: what produced a trace, so a later run can be checked against it.

Fingerprints are short sha256 digests of every model-visible text (system
prompts, strategy/persona instructions, the move tool schema). Two traces with
the same fingerprints saw the same instructions; a different fingerprint means
the benchmark condition changed, even if the configs look the same.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

# Provenance keys that describe the run, not the condition: they are expected to
# differ between runs and are never a reason to warn.
RUN_ONLY_KEYS = frozenset({"code_version", "started_at"})


def fingerprint(value: Any) -> str:
    """Short, stable sha256 digest of a string or JSON-serialisable value."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@cache
def code_version() -> str:
    """The git commit of this checkout, with "+dirty" for uncommitted changes."""
    root = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}+dirty" if dirty else sha


def tool_fingerprint(*, strict: bool, message_channel: bool) -> str:
    """Fingerprint of the move tool's shape, independent of any one scenario.

    The tool is built from the scenario's issues; building it for one fixed
    reference issue captures what the code generates (field names, descriptions,
    enums) without making runs over different scenario sets look different.
    """
    from rfq_bench.agent.prompts import build_move_tool
    from rfq_bench.core.contracts import Issue

    reference = [Issue(name="price", unit="EUR", values=[1, 2])]
    return fingerprint(build_move_tool(reference, strict=strict, message_channel=message_channel))


def run_provenance() -> dict[str, str]:
    """The run-level entries: code version and start time (UTC, ISO 8601)."""
    return {
        "code_version": code_version(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def condition_differences(
    a: list[dict[str, str]], b: list[dict[str, str]] | None = None
) -> list[str]:
    """Condition fingerprints that differ, as readable warnings.

    With ``b``: keys whose values differ between the two groups of traces. Without
    it: keys with more than one value within ``a``. Keys missing on either side
    (older traces) are skipped; run-only keys never warn.
    """

    def values(group: list[dict[str, str]], key: str) -> set[str]:
        return {p[key] for p in group if key in p}

    keys = sorted({k for p in (a + (b or [])) for k in p} - RUN_ONLY_KEYS)
    out: list[str] = []
    for key in keys:
        va = values(a, key)
        if b is None:
            if len(va) > 1:
                out.append(f"{_label(key)} changed within the run ({len(va)} versions)")
            continue
        vb = values(b, key)
        if va and vb and va != vb:
            out.append(f"{_label(key)} differs between the runs")
    return out


def _label(key: str) -> str:
    """'system_prompt_seller' -> 'seller system prompt'; 'instruction_buyer.neutral' ->
    'buyer instruction (neutral)'; 'tool_schema' -> 'move tool schema'."""
    if key.startswith("system_prompt_"):
        return f"{key.removeprefix('system_prompt_')} system prompt"
    if key.startswith("instruction_"):
        role, _, arm = key.removeprefix("instruction_").partition(".")
        return f"{role} instruction" + (f" ({arm})" if arm else "")
    if key == "tool_schema":
        return "move tool schema"
    return key.replace("_", " ")
