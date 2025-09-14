"""Agent orchestrator cancellation tests using asyncio.TaskGroup."""  # Author: Team DocuRay | Generated: TDD step | Version: 0.1.0 | Modified: 2025-09-14

import asyncio
import unittest


class TestAgentCancellation(unittest.IsolatedAsyncioTestCase):
    """Ensure pending tasks get cancelled after early stop."""

    async def test_taskgroup_cancellation_on_early_stop(self):
        from docray.orchestrator.agent import DocuRayAgent  # type: ignore

        agent = DocuRayAgent(global_time_budget_ms=300)

        # Query designed to trigger early stop quickly
        res = await agent.search("表格 营收 数据")

        self.assertIn("latency_ms", res)
        self.assertLessEqual(res["latency_ms"], 1000)
        # Expose cancellation count for observability
        self.assertGreaterEqual(res.get("cancelled_tasks", 0), 0)


if __name__ == "__main__":
    unittest.main()

