"""
Core module for QVERIFY.

This module provides the fundamental data structures and types used throughout
the QVERIFY framework.
"""

from qverify.core.types import (
    VerificationStatus,
    SynthesisStatus,
    QubitBasis,
    GateType,
    QubitId,
    QuantumState,
    CounterExample,
    VerificationResult,
    SynthesisResult,
    ProgramLocation,
    VerificationCondition,
)
from qverify.core.quantum_program import (
    ProgramLanguage,
    Qubit,
    Gate,
    Loop,
    Conditional,
    Function,
    QuantumProgram,
)
from qverify.core.specification import (
    Condition,
    AtomicCondition,
    CompoundCondition,
    QuantumCondition,
    Precondition,
    Postcondition,
    Invariant,
    Specification,
)
from qverify.core.quantum_state import (
    BasisState,
    QuantumStatePredicate,
    BasisPredicate,
    EntanglementPredicate,
    AmplitudePredicate,
    ProbabilityPredicate,
    SuperpositionPredicate,
    NegatedPredicate,
    ConjunctionPredicate,
    DisjunctionPredicate,
    parse_predicate,
)

__all__ = [
    # Types
    "VerificationStatus",
    "SynthesisStatus",
    "QubitBasis",
    "GateType",
    "QubitId",
    "QuantumState",
    "CounterExample",
    "VerificationResult",
    "SynthesisResult",
    "ProgramLocation",
    "VerificationCondition",
    # Program
    "ProgramLanguage",
    "Qubit",
    "Gate",
    "Loop",
    "Conditional",
    "Function",
    "QuantumProgram",
    # Specification
    "Condition",
    "AtomicCondition",
    "CompoundCondition",
    "QuantumCondition",
    "Precondition",
    "Postcondition",
    "Invariant",
    "Specification",
    # Quantum State
    "BasisState",
    "QuantumStatePredicate",
    "BasisPredicate",
    "EntanglementPredicate",
    "AmplitudePredicate",
    "ProbabilityPredicate",
    "SuperpositionPredicate",
    "NegatedPredicate",
    "ConjunctionPredicate",
    "DisjunctionPredicate",
    "parse_predicate",
]
