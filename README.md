# Multi-Tool Orchestrator Agent

Project 04 in a production-agent-skills series: an agent that orchestrates a dynamic fleet of tools — capability-based routing, permission-scoped execution with bounded replan on denial, parallel dependency-aware batching, and explicit conflict resolution when overlapping tools disagree.

## What it does

- **Capability routing, not a dispatch table.** Tools register themselves at runtime via a `@tool_def` decorator carrying capability tags (`weather.current`, `currency.convert`, ...), a permission scope, a priority, and a timeout. The planner LLM emits capabilities, never tool names; the router resolves capability → currently-registered tool(s) fresh from the registry on every request.
- **Permission scoping as planner input, not a crash.** Each tool declares a `read` / `network` / `write` scope. A denied capability produces a structured `Denial` (tool, capability, scope) instead of an error; if a subtask has zero runnable candidates, the graph routes back to the planner exactly once with the denials appended to its prompt — bounded to one replan so a small model can't retry a denied capability forever.
- **Parallel, fault-isolated execution.** Subtasks form a dependency DAG (`depends_on`); the executor topologically batches them and runs each batch with `asyncio.gather`, with each call individually wrapped in its own `try/except` + `asyncio.wait_for` so one tool's failure or timeout doesn't take its batch mates down.
- **Explicit conflict resolution.** The router can call every candidate for a capability (default), and when results diverge, one of three policies adjudicates: `priority` (deterministic, manifest-configured), `freshest` (by payload timestamp), or `llm_adjudicate` (model picks and explains, with hallucinated-tool-name rejection and fallback).
- **Live, not snapshotted, registry.** `registry.capability_catalog()` (planning) and `registry.find_by_capability()` (routing) both read the registry live on every call — a tool registered mid-session is routable on the very next request, no restart or cache invalidation.

## Quick start

```
# with a local Ollama server running
ollama pull qwen3.5:9b
uv sync
uv run pytest                 # 40 tests, ~1s, zero network
uv run pytest -m integration  # 2 live end-to-end requests against qwen3.5:9b — requires a local Ollama instance serving that model, several minutes

uv run orc tools               # live registry: capabilities, scopes, priorities
uv run orc run "Compare the weather in Tokyo and Paris, convert 100 USD to JPY, save a note"
uv run orc run "..." --scope read --resolve-policy llm_adjudicate
uv run orc run "..." --sequential   # wall-time baseline, no parallel batching
uv run orc demo                 # full five-step walkthrough, one command
```

## Learnings

- **Rich markup injection ate LLM output.** Free-text answers/rationales from the model can contain literal square brackets, and Rich's console markup parser silently swallowed them — verified in practice: a resolution rationale printed to the terminal with its policy name deleted out of the middle of the sentence. Fix was to render all dynamic/LLM-sourced text with markup disabled rather than trying to escape it upstream.
- **A bounded replan budget matters more than the replan itself.** An open-ended "retry until it works" loop on permission denial will, with a 9B model, sometimes retry the same denied capability indefinitely. Capping it at one replan makes the worst case a single wasted planning call instead of a hang, and a capability that no tool exists for at all (a hallucinated capability) is recorded as an honest failure rather than fed into the replan loop, since retrying doesn't fix a hallucination.
- **Small-model planning has a real ceiling, and the demo had to be built around it.** On a compound request, qwen3.5:9b doesn't reliably keep an optional write-scoped subtask (a summary note) in the plan, independent of whether that scope is even granted — a planning-quality trait, not an orchestrator bug. The denial→replan path is only reliably exercised with a second, single-purpose write request ("send an email confirming the trip"), which is what `traces/demo_permission_denial_replan.json` captures: a live run where `email.send` is denied, zero alternative capabilities exist, and the agent returns an honest "I cannot do this" instead of failing or hallucinating success.
- **`priority` and `freshest` are tuned to disagree on purpose.** The two built-in weather mocks have the higher-priority provider also be the stalest one, so the two deterministic resolution policies produce different answers on the same input — a deliberate, realistic tradeoff rather than a toy example where they'd always agree.
- **`llm_adjudicate` needs a hallucination guard.** Asking the model to name which candidate tool it trusts occasionally gets a tool name back that wasn't actually a candidate; the resolver rejects that output and falls back to the first candidate (recording that it did so) rather than propagating a reference to a tool that was never called.
- **Parallel batching was measured, not assumed.** A live run with 5 concurrent calls in a batch showed a 3.88x speedup versus the summed sequential-estimate baseline; `tests/test_executor.py` backs this with a timestamp-overlap assertion (not just that `asyncio.gather` was invoked) plus a fault-isolation test that injects a raising tool into a batch and asserts its neighbor still succeeds.
- **Dynamic registration was verified live, not just unit-tested.** A third weather provider (`weather_c`) registered mid-process joined an existing two-way priority conflict and won under the `priority` policy on the very next request, with no restart and no code path touched outside the single call to `registry.register()`.

See `readme.html` for the full write-up, architecture diagram, and permission/conflict-resolution tables.
