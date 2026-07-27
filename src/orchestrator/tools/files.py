"""File reader tool, sandboxed to CONFIG.data_dir. A planner-supplied path
is untrusted input; this resolves it and refuses anything that escapes the
sandbox directory (path traversal, absolute paths elsewhere on disk)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from orchestrator.config import CONFIG
from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def


def _resolve_sandboxed(path: str) -> Path:
    base = Path(CONFIG.data_dir).resolve()
    candidate = (base / path).resolve()
    if base not in candidate.parents and candidate != base:
        raise PermissionError(f"path '{path}' escapes the sandboxed data directory")
    return candidate


@tool_def(
    name="files",
    description="Reads a text file from the local sandboxed data directory.",
    capabilities=["file.read"],
    scope=ScopeClass.READ,
    priority=1,
    timeout_s=3.0,
    param_schema={"path": "file name relative to the sandboxed data directory"},
)
async def files(path: str) -> dict:
    await asyncio.sleep(0)
    try:
        resolved = _resolve_sandboxed(path)
    except PermissionError as exc:
        return {"path": path, "error": str(exc)}
    if not resolved.exists() or not resolved.is_file():
        return {"path": path, "error": f"no such file: '{path}'"}
    return {"path": path, "content": resolved.read_text(encoding="utf-8")}
