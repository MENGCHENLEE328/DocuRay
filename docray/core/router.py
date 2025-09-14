"""Fast query router for DocuRay."""  # Author: Team DocuRay | Generated: TDD impl | Version: 0.1.0 | Modified: 2025-09-14

from __future__ import annotations

import time
from typing import Dict


class QueryRouter:
    """Lightweight intent detection and routing."""

    def __init__(self) -> None:
        self._route_cache: Dict[int, Dict] = {}
        self._patterns = self._compile_patterns()
        self._model = None

    def load_models(self) -> None:  # Load tiny ONNX if available
        self._model = None  # Placeholder for future onnxruntime session

    def analyze_query(self, query: str) -> Dict:
        start = time.perf_counter()
        key = hash(query)
        if key in self._route_cache:
            out = self._route_cache[key]
            out["latency_ms"] = (time.perf_counter() - start) * 1000
            return out

        intent = self._quick_classify(query)
        route = self._route_for_intent(intent)

        result = {
            "query": query,
            "intent": intent,
            "entities": {"keywords": query.split()[:5]},
            "route": route,
            "complexity": self._estimate_complexity(query),
            "latency_ms": (time.perf_counter() - start) * 1000,
        }
        self._route_cache[key] = result
        return result

    def _quick_classify(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ("文件名", "路径", "位置", ".pdf", ".doc", ".md")):
            return "exact_file"
        if any(k in q for k in ("函数", "代码", "class", "def")):
            return "code_search"
        if any(k in q for k in ("表格", "数据", "统计")):
            return "table_search"
        return "semantic_search"

    def _route_for_intent(self, intent: str) -> str:
        mapping = {
            "exact_file": "fast_file_search",
            "code_search": "ast_analysis_path",
            "table_search": "table_extraction_path",
            "semantic_search": "vector_search_path",
        }
        return mapping.get(intent, "vector_search_path")

    def _estimate_complexity(self, query: str) -> float:
        return min(len(query) / 100.0, 1.0)

    def _compile_patterns(self):  # Reserved for future regex/onnx
        return {}

