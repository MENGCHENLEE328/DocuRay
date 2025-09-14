"""Early-stop tests for fixed return signature and thresholds."""  # Author: Team DocuRay | Generated: TDD step | Version: 0.1.0 | Modified: 2025-09-14

import unittest


class TestEarlyStop(unittest.TestCase):
    """Ensure should_stop returns (stop, confidence, reason)."""

    def setUp(self):
        from docray.core.early_stop import EarlyStoppingEngine  # type: ignore
        self.es = EarlyStoppingEngine()

    def test_return_signature_and_reason(self):
        stop, conf, reason = self.es.should_stop(
            partial_results=[{"score": 0.9}, {"score": 0.2}],
            elapsed_ms=100,
            query_complexity=0.2,
            time_budget=1000,
        )
        self.assertIsInstance(stop, bool)
        self.assertIsInstance(conf, float)
        self.assertIsInstance(reason, str)
        self.assertNotEqual(reason.strip(), "")

    def test_triggers_on_time_pressure(self):
        stop, conf, reason = self.es.should_stop(
            partial_results=[{"score": 0.2}, {"score": 0.19}],
            elapsed_ms=1200,
            query_complexity=0.8,
            time_budget=1300,
        )
        self.assertTrue(stop)
        self.assertIn("time", reason.lower())


if __name__ == "__main__":
    unittest.main()

