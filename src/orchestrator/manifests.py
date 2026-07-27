"""Pydantic contracts shared across the orchestrator.

``ToolManifest`` is the static description of a tool (what it's called,
what it can do, what it costs to run, what permission it needs).
The other models here describe the *dynamic* artifacts that flow through
the graph: a planner's subtasks, a router's candidate list, a single tool
call's outcome, and the conflicts/resolutions the resolver produces.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ScopeClass(str, Enum):
    """Coarse permission classes a session can grant or withhold.

    READ    - queries public/local information, no side effects.
    WRITE   - mutates state the user cares about (files, outbound messages).
    NETWORK - reaches an external, less-trusted service.
    """

    READ = "read"
    WRITE = "write"
    NETWORK = "network"

    @classmethod
    def parse_set(cls, raw: str) -> set["ScopeClass"]:
        """Parse a comma-separated CLI value like 'read,network'."""
        raw = raw.strip()
        if not raw:
            return set()
        return {cls(part.strip().lower()) for part in raw.split(",") if part.strip()}


ALL_SCOPES: set[ScopeClass] = {ScopeClass.READ, ScopeClass.WRITE, ScopeClass.NETWORK}


class ToolManifest(BaseModel):
    """Static, registry-held description of one callable tool."""

    name: str
    description: str
    capabilities: list[str] = Field(min_length=1)
    scope: ScopeClass
    priority: int = 1
    """Higher wins ties under the 'priority' router/resolver policy."""
    cost_hint: float = 1.0
    """Relative cost/latency hint, arbitrary units. Purely informational today."""
    timeout_s: float = 10.0
    param_schema: dict[str, str] = Field(default_factory=dict)
    """capability arg name -> one-line human description, shown to the planner."""


class Subtask(BaseModel):
    """One planner-emitted unit of work, tagged by capability - never by tool name."""

    id: str
    capability: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    rationale: str = ""


class Plan(BaseModel):
    subtasks: list[Subtask]
    notes: str = ""
    """Free-text planner commentary, e.g. what it dropped in a replan and why."""


class Denial(BaseModel):
    subtask_id: str
    capability: str
    tool: str
    scope_required: ScopeClass
    reason: str


class ToolCall(BaseModel):
    """A single (subtask, tool) pairing selected by the router, ready to execute."""

    subtask_id: str
    capability: str
    tool: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    subtask_id: str
    capability: str
    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    observed_at: float = Field(default_factory=time.time)
    """Freshness timestamp: when the underlying data was produced. Tools that
    simulate stale data (e.g. a slow-to-update weather feed) override this;
    everything else defaults to call completion time."""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


class BatchReport(BaseModel):
    batch_index: int
    call_count: int
    wall_time_s: float
    sequential_estimate_s: float

    @property
    def speedup(self) -> float:
        if self.wall_time_s <= 0:
            return 1.0
        return round(self.sequential_estimate_s / self.wall_time_s, 2)


class Conflict(BaseModel):
    subtask_id: str
    capability: str
    results: list[ToolResult]


class Resolution(BaseModel):
    subtask_id: str
    capability: str
    policy: str
    chosen_tool: str
    rationale: str


class ExecutionReport(BaseModel):
    """Everything the aggregator and the CLI need to explain what happened."""

    plan: Plan
    denials: list[Denial] = Field(default_factory=list)
    unroutable: list[str] = Field(default_factory=list)
    """Capabilities the planner emitted that have zero registered tools -
    an "honest failure" path distinct from a permission denial: nothing to
    gate, there's simply no candidate. Not replanned (see router.py)."""
    replans: int = 0
    batches: list[BatchReport] = Field(default_factory=list)
    results: list[ToolResult] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    resolutions: list[Resolution] = Field(default_factory=list)
    answer: str = ""

    @property
    def total_wall_time_s(self) -> float:
        return round(sum(b.wall_time_s for b in self.batches), 3)

    @property
    def total_sequential_estimate_s(self) -> float:
        return round(sum(b.sequential_estimate_s for b in self.batches), 3)
