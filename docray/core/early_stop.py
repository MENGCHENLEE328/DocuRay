"""Early stopping engine with fixed return signature."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

import math
from typing import List, Dict, Tuple


class EarlyStoppingEngine:
    """Computes stop decision with confidence and reason."""

    def __init__(self) -> None:
        from docray.utils.config import load_yaml
        cfg = load_yaml("configs/early_stop.yaml")
        self._hist: List[float] = []
        self._conf_bias: float = float(cfg.get("confidence_bias", 0.0))
        self._w_complexity: float = float(cfg.get("complexity_weight", 0.2))
        self._w_time: float = float(cfg.get("time_pressure_weight", 0.1))
        self._default_budget: float = float(cfg.get("time_budget_ms", 1800))

    def should_stop(
        self,
        partial_results: List[Dict],
        elapsed_ms: float,
        query_complexity: float = 0.5,
        time_budget: float = 1800.0,
    ) -> Tuple[bool, float, str]:
        if not partial_results:
            return False, 0.0, "insufficient-evidence"

        scores = [float(r.get("score", 0.0)) for r in partial_results[:10]]
        if not scores:
            return False, 0.0, "no-scores"

        ssum = sum(scores) or 1e-6
        norm = [s / ssum for s in scores]

        top = norm[0]
        gap = (norm[0] - norm[1]) if len(norm) > 1 else 1.0
        entropy = -sum(s * math.log(s + 1e-6) for s in norm)
        entropy_max = math.log(len(norm)) if len(norm) > 1 else 1.0

        conf = 0.4 * top + 0.3 * min(gap * 3.0, 1.0) + 0.3 * (1.0 - (entropy / entropy_max))
        conf += self._conf_bias

        budget = time_budget or self._default_budget
        time_pressure = elapsed_ms / max(budget, 1.0)
        dynamic_threshold = 0.9 - self._w_complexity * query_complexity - self._w_time * time_pressure

        hist_adj = 0.0
        if self._hist:
            hist_adj = 0.3 * (sum(self._hist[-5:]) / min(len(self._hist[-5:]), 5))
            conf = 0.7 * conf + hist_adj

        stop_by_conf = conf > dynamic_threshold
        stop_by_time = elapsed_ms >= 0.9 * time_budget
        stop = bool(stop_by_conf or stop_by_time)

        reason = (
            "confidence-threshold" if stop_by_conf else ("time-budget" if stop_by_time else "continue")
        )

        self._hist.append(conf)
        return stop, float(conf), reason
