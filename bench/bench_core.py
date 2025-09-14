"""Micro benchmarks for router and early-stop."""  # Author: Team DocuRay | Generated: bench seed | Version: 0.1.0 | Modified: 2025-09-14

import time
from statistics import median

from docray.core.router import QueryRouter
from docray.core.early_stop import EarlyStoppingEngine


def bench_router(n: int = 1000) -> float:  # return median us
    r = QueryRouter(); r.load_models()
    qs = ["查找文件 report.pdf", "函数 handleError", "表格 营收 数据", "公司文化"]
    times = []
    for i in range(n):
        q = qs[i % len(qs)]
        t0 = time.perf_counter(); r.analyze_query(q); t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return median(times)


def bench_early_stop(n: int = 1000) -> float:
    es = EarlyStoppingEngine()
    pr = [{"score": 0.9 - i * 0.1} for i in range(5)]
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); es.should_stop(pr, 100.0, 0.3, 1800.0); t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return median(times)


if __name__ == "__main__":
    mr = bench_router()
    me = bench_early_stop()
    print(f"Router median latency: {mr:.1f} us")
    print(f"Early-stop median latency: {me:.1f} us")

