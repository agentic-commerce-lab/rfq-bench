"""Run-usage summary: prompt-cache hit rate and A2A message-channel activity.

Printed by ``rfq-bench report`` below the score tables. Each section appears only
when the traces carry the data (older traces predate cached-token and message
recording, and scripted runs have neither), so it never adds noise.
"""

from __future__ import annotations

from collections import defaultdict

from rfq_bench.core.contracts import Trace


def usage_summary(traces: list[Trace]) -> list[str]:
    """Human-readable lines; empty when there is nothing to report."""
    lines: list[str] = []
    lines += _policy_lines(traces)
    lines += _cache_lines(traces)
    lines += _message_lines(traces)
    return lines


def _policy_lines(traces: list[Trace]) -> list[str]:
    """How often the harness had to correct an invalid move, by role, kind and arm.

    A move's arm is the scored strategy when the target agent made it, else the
    opponent (the buyer persona in A2A). Only LLM sides can trigger a constraint.
    """
    turns: dict[str, int] = defaultdict(int)
    fired: dict[str, int] = defaultdict(int)
    kinds: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_arm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    withheld = 0
    for t in traces:
        for s in t.steps:
            if s.error:
                continue
            turns[s.party] += 1
            if not s.adjusted:
                continue
            fired[s.party] += 1
            kinds[s.party][s.adjust_kind or "other"] += 1
            arm = t.strategy if s.party == t.target_role else t.opponent
            by_arm[s.party][arm] += 1
            withheld += int(s.message_withheld)
    if not sum(fired.values()):
        return []
    lines = ["", "Policy constraints fired (invalid moves the harness corrected)"]
    for role in sorted(turns):
        if not fired[role]:
            lines.append(f"  {role:<7} 0 of {turns[role]} turns")
            continue
        kind = ", ".join(f"{k} {n}" for k, n in sorted(kinds[role].items()))
        arms = ", ".join(f"{a} {n}" for a, n in sorted(by_arm[role].items(), key=lambda x: -x[1]))
        share = 100 * fired[role] / turns[role]
        head = f"  {role:<7} {fired[role]} of {turns[role]} turns ({share:.0f}%)"
        lines.append(f"{head}: {kind} · by arm: {arms}")
    if withheld:
        lines.append(f"  messages withheld from the opponent: {withheld}")
    return lines


def _cache_lines(traces: list[Trace]) -> list[str]:
    prompt: dict[str, int] = defaultdict(int)
    cached: dict[str, int] = defaultdict(int)
    for t in traces:
        for role, n in t.prompt_tokens_by_role.items():
            prompt[role] += n
            cached[role] += t.cached_tokens_by_role.get(role, 0)
    total = sum(prompt.values())
    if not total:
        return []
    hit = sum(cached.values()) / total
    by_role = ", ".join(
        f"{role} {100 * cached[role] / prompt[role]:.0f}%"
        for role in sorted(prompt)
        if prompt[role]
    )
    return [
        "",
        f"Prompt cache: {100 * hit:.0f}% of {total:,} prompt tokens served from cache ({by_role})",
    ]


def _message_lines(traces: list[Trace]) -> list[str]:
    turns: dict[str, int] = defaultdict(int)
    sent: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    cut: dict[str, int] = defaultdict(int)
    for t in traces:
        if t.mode != "a2a":
            continue
        for s in t.steps:
            if s.error:
                continue
            turns[s.party] += 1
            if s.message:
                sent[s.party] += 1
                chars[s.party] += len(s.message)
                cut[s.party] += int(s.message_truncated)
    if not sum(sent.values()):
        return []
    lines = ["", "Messages (A2A channel)"]
    for role in sorted(turns):
        if not turns[role]:
            continue
        share = 100 * sent[role] / turns[role]
        mean = chars[role] / sent[role] if sent[role] else 0.0
        note = f", {cut[role]} truncated" if cut[role] else ""
        lines.append(
            f"  {role:<7} sent on {share:.0f}% of {turns[role]} turns, mean {mean:.0f} chars{note}"
        )
    return lines
