"""Note-writer tool - a genuine local side effect, gated behind WRITE scope.
Writes a markdown file into CONFIG.notes_dir."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

from orchestrator.config import CONFIG
from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "note"


@tool_def(
    name="notes",
    description="Writes a markdown note to local disk.",
    capabilities=["note.write"],
    scope=ScopeClass.WRITE,
    priority=1,
    timeout_s=3.0,
    param_schema={"title": "short note title", "content": "note body text"},
)
async def notes(title: str, content: str) -> dict:
    await asyncio.sleep(0)
    out_dir = Path(CONFIG.notes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_slugify(title)}_{time.strftime('%Y%m%d-%H%M%S')}.md"
    out_path = out_dir / filename
    out_path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return {"title": title, "path": str(out_path)}
