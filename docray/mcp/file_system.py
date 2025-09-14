"""MCP stub for filesystem queries respecting deadlines."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

import asyncio
from typing import Dict, Any


class FileSystemMCP:
    """Minimal async adapter."""

    async def call(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        delay = float(params.get("delay_ms", 80)) / 1000.0
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise

        if action == "find_files":
            return {
                "files": [
                    {"path": "/docs/file1.pdf", "size": 1024},
                    {"path": "/docs/file2.md", "size": 512},
                ]
            }
        return {}

