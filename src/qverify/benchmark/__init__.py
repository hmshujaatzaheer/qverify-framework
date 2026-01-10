"""
Benchmark module for QVERIFY.

This module provides QVerifyBench - a comprehensive benchmark for evaluating
LLMs on quantum program verification tasks.
"""

from qverify.benchmark.qverifybench import (
    QVerifyBench,
    BenchmarkTier,
    BenchmarkProgram,
    BenchmarkResults,
    EvaluationResult,
)
from qverify.benchmark.metrics import (
    SpecificationMetrics,
    PerformanceMetrics,
    TierMetrics,
    EvaluationReport,
    compute_specification_similarity,
)

__all__ = [
    # QVerifyBench
    "QVerifyBench",
    "BenchmarkTier",
    "BenchmarkProgram",
    "BenchmarkResults",
    "EvaluationResult",
    # Metrics
    "SpecificationMetrics",
    "PerformanceMetrics",
    "TierMetrics",
    "EvaluationReport",
    "compute_specification_similarity",
]
