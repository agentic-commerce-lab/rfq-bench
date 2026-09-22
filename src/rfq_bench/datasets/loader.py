"""Load and validate scenario datasets from JSON.

A dataset file is a JSON object with a top-level ``scenarios`` list, each entry
matching the :class:`Scenario` schema. Validation is pydantic's plus a ZOPA
sanity check that flags degenerate scenarios early.
"""

from __future__ import annotations

import json
from pathlib import Path

from rfq_bench.core.contracts import Scenario, ScenarioKind
from rfq_bench.core.zopa import compute_zopa, is_degenerate


def load_scenario_file(path: str | Path) -> list[Scenario]:
    """Parse one dataset file into validated scenarios."""
    raw = json.loads(Path(path).read_text())
    entries = raw["scenarios"] if isinstance(raw, dict) else raw
    scenarios = [Scenario.model_validate(entry) for entry in entries]
    _sanity_check(scenarios)
    return scenarios


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Load and concatenate every ``*.json`` dataset in ``directory`` (sorted)."""
    directory = Path(directory)
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        for scenario in load_scenario_file(path):
            if scenario.id in seen:
                raise ValueError(f"duplicate scenario id {scenario.id!r} in {path}")
            seen.add(scenario.id)
            scenarios.append(scenario)
    return scenarios


def _sanity_check(scenarios: list[Scenario]) -> None:
    for s in scenarios:
        for role in ("buyer", "seller"):
            if is_degenerate(s, role):
                raise ValueError(f"scenario {s.id!r} is degenerate for {role} (ideal == BATNA)")
        zopa = compute_zopa(s)
        if s.kind == ScenarioKind.no_zopa_diagnostic and zopa.exists:
            raise ValueError(f"scenario {s.id!r} is tagged no_zopa but a ZOPA exists")
        if s.kind != ScenarioKind.no_zopa_diagnostic and not zopa.exists:
            raise ValueError(f"scenario {s.id!r} has no ZOPA but is not tagged no_zopa_diagnostic")
