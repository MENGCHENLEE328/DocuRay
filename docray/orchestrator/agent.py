"""Agent orchestrator with TaskGroup cancellation and deadline propagation."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, List

from docray.core.router import QueryRouter
from docray.core.early_stop import EarlyStoppingEngine
from docray.mcp.vector_db import VectorDBMCP
from docray.mcp.file_system import FileSystemMCP
from docray.utils.logging import get_logger


class DocuRayAgent:
    """Minimal orchestrator for tests."""

    def __init__(self, global_time_budget_ms: int = 800) -> None:
        self.router = QueryRouter()
        self.early = EarlyStoppingEngine()
        self.vec = VectorDBMCP()
        self.fs = FileSystemMCP()
        self.log = get_logger("docray.agent")
        self.global_budget = float(global_time_budget_ms)

    async def search(self, query: str) -> Dict[str, Any]:
        start = time.perf_counter()
        route = self.router.analyze_query(query)
        deadline_ms = self.global_budget

        partial: List[Dict[str, Any]] = []
        cancelled = 0

        async def run_vec():
            out = await self.vec.call("search", {"delay_ms": 200})
            for i, r in enumerate(out.get("results", [])):
                partial.append({"id": r["id"], "score": r["score"], "channel": "vector", "rank": i})

        async def run_fs():
            out = await self.fs.call("find_files", {"delay_ms": 300})
            for i, f in enumerate(out.get("files", [])):
                # synthetic score for files
                partial.append({"id": f["path"], "score": 0.3, "channel": "meta", "rank": i})

        async with asyncio.TaskGroup() as tg:
            t1 = tg.create_task(run_vec())
            t2 = tg.create_task(run_fs())

            # polling loop to decide early stop
            while True:
                elapsed = (time.perf_counter() - start) * 1000
                stop, _conf, reason = self.early.should_stop(partial, elapsed, route.get("complexity", 0.5), deadline_ms)
                if stop:
                    for task in (t1, t2):
                        if not task.done():
                            task.cancel()
                            cancelled += 1
                    break
                if all(task.done() for task in (t1, t2)):
                    break
                await asyncio.sleep(0.01)

        latency = (time.perf_counter() - start) * 1000
        return {"latency_ms": latency, "cancelled_tasks": cancelled, "route": route["route"], "results": partial}

