"""MCP stub for vector DB operations respecting deadlines."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

import asyncio
from typing import Dict, Any


class VectorDBMCP:
    """Minimal async adapter."""

    async def call(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        delay = float(params.get("delay_ms", 120)) / 1000.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise

        if action == "search":
            return {
                "results": [
                    {"id": f"doc_{i}", "score": 0.9 - i * 0.1} for i in range(5)
                ]
            }
        return {}

