# Multi-Tool Orchestrator Agent

Project 04 in a production-agent engineering series: an agent that routes over a dynamic tool fleet using capabilities rather than tool names, applies scope checks before execution, batches independent work in parallel, and resolves conflicting provider results explicitly.

## What it does

- **Capability routing rather than a dispatch table.** Tools register at runtime through a `@tool_def` decorator with capability tags, an execution scope, priority, and timeout. The planner requests capabilities; the router resolves them against the live registry.
- **Structured permission denial.** Every tool declares a `read`, `network`, or `write` scope. A denied capability becomes a typed `Denial` that the planner can see. If no runnable candidate remains, the graph permits exactly one replan instead of retrying indefinitely.
- **Parallel, fault-isolated execution.** Subtasks form a dependency DAG. Ready tasks run through `asyncio.gather`, while each call gets an independent timeout and exception boundary so one provider cannot take down its batch mates.
- **Explicit conflict resolution.** In `ALL` mode the router can call every provider for a capability. Divergent results are resolved by manifest priority, data freshness, or LLM adjudication. A failed adjudicator or hallucinated tool name now falls back to deterministic manifest priority—not whichever result happened to appear first.
- **Live registry semantics.** Planning and routing read the registry on every request, so a provider registered mid-process is eligible immediately without restarting or invalidating a cache.

## Quick start

```bash
ollama pull qwen3.5:9b
uv sync

# Fast unit and mocked-model suite
uv run pytest

# Live end-to-end tests
uv run pytest -m integration

uv run orc tools
uv run orc run "Compare the weather in Tokyo and Paris, convert 100 USD to JPY, save a note"
uv run orc run "..." --scope read --resolve-policy llm_adjudicate
uv run orc run "..." --sequential
uv run orc demo
```

## Authorization and trust boundary

The `read` / `network` / `write` values are an orchestration policy demonstration, not a complete authorization system. The caller supplies allowed scopes; this project does not authenticate a human or workload identity, issue short-lived credentials, enforce resource-level permissions, protect secrets, or prevent a tool implementation from acting outside its declared manifest. In production, the executor—not the model and not a manifest string—must enforce least privilege through real identity, policy, network, and runtime controls.

Tool payloads are also untrusted. An LLM adjudicator can be influenced by malicious content inside a provider response, and “freshest” is only useful when timestamps are trustworthy and comparable. High-stakes resolution should combine authenticated provenance, deterministic policy, domain-specific validation, and human review rather than relying on a model to decide which source is true.

## Learnings

- **Rich markup consumed model text.** Free-form rationales containing square brackets were interpreted by the terminal renderer. Dynamic model output is now printed with markup disabled.
- **A bounded replan budget matters more than replanning alone.** A small model may repeatedly request a denied capability. One permitted replan converts an open-ended loop into one bounded extra planning call.
- **Small-model planning has a real ceiling.** The model did not reliably preserve an optional write subtask in a compound plan, so the denial demonstration uses a focused write request and reports that limitation instead of hiding it.
- **Priority and freshness deliberately disagree in the fixtures.** The higher-priority weather provider is also the older one, making the policy trade-off observable.
- **LLM adjudication needs a deterministic fallback.** Invalid candidate names and transport failures now select the highest-priority registered candidate and record why the fallback occurred.
- **Parallelism was measured rather than inferred.** A live five-call batch achieved a 3.88× speedup over the summed sequential estimate; tests assert temporal overlap and neighbor survival when one tool raises.
- **Dynamic registration was verified live.** A third weather provider registered mid-process participated in the next conflict and won under priority without a restart.

See [`readme.html`](readme.html) for the full architecture, permission matrix, conflict-resolution table, and captured traces.
