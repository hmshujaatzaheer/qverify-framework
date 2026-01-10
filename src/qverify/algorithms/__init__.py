"""
Algorithms module for QVERIFY.

This module provides the core algorithms for specification synthesis,
predicate learning, invariant synthesis, and specification repair.
"""

from qverify.algorithms.spec_synth import (
    QuantumSpecSynth,
    ProgramAnalysis,
    TypeInference,
)
from qverify.algorithms.predicate_learning import (
    LearnQuantumPredicate,
    ExecutionTrace,
    PredicateCandidate,
    learn_predicate_from_states,
)
from qverify.algorithms.invariant_synth import (
    SynthesizeQuantumInvariant,
    LoopAnalysis,
)
from qverify.algorithms.spec_repair import (
    RepairSpecification,
    FailureDiagnosis,
    repair_with_counterexample,
)

__all__ = [
    # Specification synthesis
    "QuantumSpecSynth",
    "ProgramAnalysis",
    "TypeInference",
    # Predicate learning
    "LearnQuantumPredicate",
    "ExecutionTrace",
    "PredicateCandidate",
    "learn_predicate_from_states",
    # Invariant synthesis
    "SynthesizeQuantumInvariant",
    "LoopAnalysis",
    # Specification repair
    "RepairSpecification",
    "FailureDiagnosis",
    "repair_with_counterexample",
]
