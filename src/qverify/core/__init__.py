"""Core data structures for QVERIFY."""

from qverify.core.quantum_program import QuantumProgram
from qverify.core.quantum_state import (
    CNOT,
    HADAMARD,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    SWAP,
    QuantumStateAnalyzer,
    create_bell_state,
    create_ghz_state,
    is_entangled,
    state_fidelity,
    tensor_product,
)
from qverify.core.specification import (
    Invariant,
    Postcondition,
    Precondition,
    Specification,
)
from qverify.core.types import (
    BenchmarkResult,
    CounterExample,
    Gate,
    QuantumState,
    SynthesisResult,
    SynthesisStatus,
    VerificationCondition,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "QuantumProgram",
    "QuantumState",
    "QuantumStateAnalyzer",
    "create_bell_state",
    "create_ghz_state",
    "is_entangled",
    "state_fidelity",
    "tensor_product",
    "HADAMARD",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    "CNOT",
    "SWAP",
    "Gate",
    "Specification",
    "Precondition",
    "Postcondition",
    "Invariant",
    "CounterExample",
    "VerificationCondition",
    "VerificationResult",
    "VerificationStatus",
    "SynthesisResult",
    "SynthesisStatus",
    "BenchmarkResult",
]
