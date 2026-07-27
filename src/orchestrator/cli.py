"""Command-line entry point.

    orc tools
    orc run "<request>" [--scope read,write,network] [--router-mode all|priority]
                         [--resolve-policy priority|freshest|llm_adjudicate] [--sequential]
                         [--save-trace]
    orc demo
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.text import Text

from orchestrator import tools as _tools  # noqa: F401  (import side effect: registers the 7 default tools)
from orchestrator.graph import run_orchestrator
from orchestrator.manifests import ALL_SCOPES, ExecutionReport, ScopeClass
from orchestrator.registry import REGISTRY
from orchestrator.resolver import ResolvePolicy
from orchestrator.router import RouterMode

console = Console()


def _print_tools() -> None:
    table = Table(title=f"Registered tools ({len(REGISTRY)})")
    for col in ("name", "capabilities", "scope", "priority", "cost_hint", "timeout_s"):
        table.add_column(col)
    for tool in REGISTRY.list_tools():
        m = tool.manifest
        table.add_row(
            m.name, ", ".join(m.capabilities), m.scope.value, str(m.priority), str(m.cost_hint), str(m.timeout_s)
        )
    console.print(table)


def _render_report(report: ExecutionReport, *, elapsed: float | None = None, show_plan: bool = True) -> None:
    # Everything below that interpolates planner/tool/LLM-generated text
    # uses markup=False (or a Text(...) cell, for tables). Rich's markup
    # parser treats a literal '[' as the start of a style tag - free text
    # from a 9B model easily contains brackets, and letting Rich parse
    # them silently swallows chunks of the answer instead of printing it.
    if show_plan:
        table = Table(title="Subtask plan")
        for col in ("id", "capability", "args", "depends_on"):
            table.add_column(col)
        for st in report.plan.subtasks:
            table.add_row(st.id, st.capability, Text(str(st.args)), ", ".join(st.depends_on) or "-")
        console.print(table)
        if report.plan.notes:
            console.print("[italic]planner notes:[/italic]", end=" ")
            console.print(report.plan.notes, markup=False)

    if report.denials:
        console.print("[bold red]Denials[/bold red] (planner adapted around these)")
        for d in report.denials:
            console.print(f"  - {d.tool} for {d.capability}: {d.reason}", markup=False)

    if report.unroutable:
        console.print("[bold yellow]Unroutable capabilities:[/bold yellow]", end=" ")
        console.print(", ".join(report.unroutable), markup=False)

    if report.batches:
        table = Table(title="Execution batches (parallel within each batch)")
        for col in ("batch", "calls", "wall_time_s", "sequential_estimate_s", "speedup"):
            table.add_column(col)
        for b in report.batches:
            table.add_row(
                str(b.batch_index), str(b.call_count), f"{b.wall_time_s:.3f}",
                f"{b.sequential_estimate_s:.3f}", f"{b.speedup:.2f}x",
            )
        console.print(table)
        console.print(
            f"Total wall time: {report.total_wall_time_s:.3f}s  |  "
            f"sequential estimate: {report.total_sequential_estimate_s:.3f}s"
        )

    if report.conflicts:
        console.print("[bold magenta]Conflicts detected[/bold magenta]")
        for c in report.conflicts:
            detail = "; ".join(f"{r.tool}={r.data}" for r in c.results)
            console.print(f"  - {c.subtask_id} ({c.capability}): {detail}", markup=False)
        for r in report.resolutions:
            console.print(f"    resolved via policy={r.policy} -> {r.chosen_tool}: {r.rationale}", markup=False)

    console.rule("Answer")
    console.print(report.answer, markup=False)
    if elapsed is not None:
        console.print(f"[dim]total run time: {elapsed:.2f}s[/dim]")


@click.group()
def main() -> None:
    """Multi-tool orchestrator: capability routing, permission scoping,
    parallel execution, conflict resolution over a dynamic tool registry."""


@main.command("tools")
def cmd_tools() -> None:
    """List the live tool registry."""
    _print_tools()


@main.command("run")
@click.argument("request")
@click.option("--scope", default="read,write,network", show_default=True, help="Comma-separated session scope.")
@click.option(
    "--router-mode",
    type=click.Choice([m.value for m in RouterMode]),
    default=RouterMode.ALL.value,
    show_default=True,
    help="'all' calls every candidate tool per capability (surfaces conflicts); 'priority' calls only the top one.",
)
@click.option(
    "--resolve-policy",
    type=click.Choice([p.value for p in ResolvePolicy]),
    default=ResolvePolicy.PRIORITY.value,
    show_default=True,
)
@click.option(
    "--sequential", is_flag=True, default=False, help="Run each batch's calls one at a time (for wall-time comparison)."
)
@click.option("--save-trace", is_flag=True, default=False, help="Save the full execution report as JSON under traces/.")
def cmd_run(request: str, scope: str, router_mode: str, resolve_policy: str, sequential: bool, save_trace: bool) -> None:
    """Run REQUEST through the orchestrator and print the execution report."""
    session_scope = ScopeClass.parse_set(scope)
    started = time.monotonic()
    report = asyncio.run(
        run_orchestrator(
            request,
            scope=session_scope,
            router_mode=RouterMode(router_mode),
            resolve_policy=ResolvePolicy(resolve_policy),
            sequential=sequential,
        )
    )
    _render_report(report, elapsed=time.monotonic() - started)

    if save_trace:
        path = _default_trace_path(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]trace saved to {path}[/dim]")


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "run"


def _default_trace_path(request: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("traces") / f"{_slugify(request)}_{timestamp}.json"


async def _run_demo() -> None:
    multi_request = "Compare the current weather in Tokyo and Paris, convert 100 USD to JPY, and save a summary note about it."
    weather_request = "What's the current weather in Tokyo?"
    write_request = "Send an email to ops@example.com confirming the trip is booked."

    console.rule("[bold]Step 1 - live registry[/bold]")
    _print_tools()

    console.rule("[bold]Step 2 - multi-part request, full scope, parallel execution[/bold]")
    report = await run_orchestrator(multi_request, scope=ALL_SCOPES, router_mode=RouterMode.ALL, resolve_policy=ResolvePolicy.PRIORITY)
    _render_report(report)

    console.rule("[bold]Step 3 - same request under --scope read (write tools denied, planner adapts)[/bold]")
    report_ro = await run_orchestrator(multi_request, scope={ScopeClass.READ}, router_mode=RouterMode.ALL, resolve_policy=ResolvePolicy.PRIORITY)
    _render_report(report_ro)

    # A 9B planner doesn't always keep a write-scoped subtask (e.g. the note)
    # in scope for a compound request, so the run above doesn't reliably
    # exercise a real denial. This request only has one thing to do and it
    # always needs WRITE scope, so it deterministically demonstrates the
    # permission_gate -> planner replan edge.
    console.print("\n[dim]-- a request that only has a WRITE-scoped path, to guarantee a real denial+replan --[/dim]")
    report_write = await run_orchestrator(write_request, scope={ScopeClass.READ}, router_mode=RouterMode.ALL, resolve_policy=ResolvePolicy.PRIORITY)
    _render_report(report_write)

    console.rule("[bold]Step 4 - weather conflict under all three resolution policies[/bold]")
    for policy in ResolvePolicy:
        console.print(f"\n[bold]-- policy: {policy.value} --[/bold]")
        report_w = await run_orchestrator(weather_request, scope=ALL_SCOPES, router_mode=RouterMode.ALL, resolve_policy=policy)
        _render_report(report_w, show_plan=False)

    console.rule("[bold]Step 5 - register weather_c mid-session, rerun (dynamic routing)[/bold]")
    console.print("Registry before:")
    _print_tools()
    from orchestrator.tools import register_weather_c

    register_weather_c()
    console.print("Registry after registering weather_c mid-session:")
    _print_tools()
    report_c = await run_orchestrator(weather_request, scope=ALL_SCOPES, router_mode=RouterMode.ALL, resolve_policy=ResolvePolicy.PRIORITY)
    _render_report(report_c, show_plan=False)


@main.command("demo")
def cmd_demo() -> None:
    """Run the full demo script end-to-end in one process: live registry,
    a multi-part parallel request, the same request under a restricted
    scope, weather-conflict resolution under all three policies, and
    finally registering a brand-new tool mid-session and routing to it."""
    asyncio.run(_run_demo())
