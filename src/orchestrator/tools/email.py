"""Mock email sender. Never touches a real mail server - simulates a send
and returns a confirmation, so the WRITE-scope demo is safe to run
repeatedly without spamming anyone."""

from __future__ import annotations

import asyncio
import time

from orchestrator.manifests import ScopeClass
from orchestrator.registry import tool_def


@tool_def(
    name="email_mock",
    description="Sends an email (mock - no real message is transmitted).",
    capabilities=["email.send"],
    scope=ScopeClass.WRITE,
    priority=1,
    timeout_s=3.0,
    param_schema={"to": "recipient address", "subject": "subject line", "body": "message body"},
)
async def email_mock(to: str, subject: str, body: str) -> dict:
    await asyncio.sleep(0.1)
    return {
        "to": to,
        "subject": subject,
        "status": "sent (mock)",
        "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
