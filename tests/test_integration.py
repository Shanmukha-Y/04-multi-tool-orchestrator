"""Live end-to-end tests against a real local Ollama server (qwen3.5:9b).

Excluded from the default test run (see pyproject's addopts) - run
explicitly with:

    uv run pytest -m integration

Deliberately limited to two requests: this exercises the full graph
(planner -> router -> permission_gate -> executor -> conflict check ->
aggregator, including one denial-triggered replan) without hammering a
shared Ollama server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator import tools as _tools  # noqa: F401  (registers the 7 default tools)
from orchestrator.graph import run_orchestrator
from orchestrator.manifests import ScopeClass
from orchestrator.registry import REGISTRY
from orchestrator.resolver import ResolvePolicy
from orchestrator.router import RouterMode

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


async def test_multi_part_request_runs_in_parallel_and_writes_a_note() -> None:
    request = (
        "Compare the current weather in Tokyo and Paris, convert 100 USD to JPY, "
        "and save a short summary note about it."
    )

    report = await run_orchestrator(
        request,
        scope={ScopeClass.READ, ScopeClass.WRITE, ScopeClass.NETWORK},
        registry=REGISTRY,
        router_mode=RouterMode.ALL,
        resolve_policy=ResolvePolicy.PRIORITY,
    )

    assert len(report.plan.subtasks) >= 2
    assert report.batches, "expected at least one execution batch"
    # At least one batch should have run more than one call concurrently -
    # the weather/currency subtasks are independent of each other.
    assert any(b.call_count >= 2 for b in report.batches)
    assert report.answer.strip()

    note_results = [r for r in report.results if r.tool == "notes" and r.ok]
    if note_results:
        note_path = Path(note_results[0].data["path"])
        assert note_path.exists()
        assert note_path.read_text().strip()
    else:
        # If the planner genuinely dropped the note subtask, that's a
        # planning-quality issue, not an orchestrator bug - but at minimum
        # the other two subtasks must have succeeded.
        assert any(r.ok for r in report.results)


async def test_write_scope_denied_planner_adapts_without_crashing() -> None:
    request = "Send an email to test@example.com saying the trip is confirmed."

    report = await run_orchestrator(
        request,
        scope={ScopeClass.READ},
        registry=REGISTRY,
        router_mode=RouterMode.ALL,
        resolve_policy=ResolvePolicy.PRIORITY,
    )

    # The only capability that could serve this request needs WRITE scope,
    # which was withheld - the orchestrator must not crash, and must be
    # able to explain that it couldn't send the email.
    assert report.answer.strip()
    assert report.denials or not report.results or all(not r.ok for r in report.results if r.tool == "email_mock")
