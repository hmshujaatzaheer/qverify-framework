"""
Core type definitions for QVERIFY.

This module contains all the fundamental types used throughout the framework,
including verification results, status enums, and data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional, Union
import numpy as np
from numpy.typing import NDArray


class VerificationStatus(Enum):
    """Status of a verification attempt."""
    
    VALID = auto()
    """The program satisfies the specification."""
    
    INVALID = auto()
    """The program violates the specification (counterexample found)."""
    
    UNKNOWN = auto()
    """Verification could not determine validity (timeout or incomplete)."""
    
    ERROR = auto()
    """An error occurred during verification."""


class SynthesisStatus(Enum):
    """Status of a specification synthesis attempt."""
    
    SUCCESS = auto()
    """Specification was successfully synthesized."""
    
    PARTIAL = auto()
    """Partial specification synthesized (some components missing)."""
    
    FAILED = auto()
    """Synthesis failed to produce a valid specification."""
    
    TIMEOUT = auto()
    """Synthesis timed out."""


class QubitBasis(Enum):
    """Standard quantum basis states."""
    
    ZERO = "|0⟩"
    ONE = "|1⟩"
    PLUS = "|+⟩"
    MINUS = "|-⟩"
    PLUS_I = "|+i⟩"
    MINUS_I = "|-i⟩"


class GateType(Enum):
    """Types of quantum gates."""
    
    # Single qubit gates
    HADAMARD = "H"
    PAULI_X = "X"
    PAULI_Y = "Y"
    PAULI_Z = "Z"
    S_GATE = "S"
    T_GATE = "T"
    RX = "RX"
    RY = "RY"
    RZ = "RZ"
    
    # Two qubit gates
    CNOT = "CNOT"
    CZ = "CZ"
    SWAP = "SWAP"
    
    # Three qubit gates
    TOFFOLI = "CCX"
    FREDKIN = "CSWAP"
    
    # Measurement
    MEASURE = "M"


@dataclass(frozen=True)
class QubitId:
    """Identifier for a qubit in a quantum program."""
    
    name: str
    index: Optional[int] = None
    
    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name
    
    def __hash__(self) -> int:
        return hash((self.name, self.index))


@dataclass
class QuantumState:
    """Representation of a quantum state."""
    
    num_qubits: int
    amplitudes: NDArray[np.complex128]
    qubit_ids: list[QubitId] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        expected_size = 2 ** self.num_qubits
        if len(self.amplitudes) != expected_size:
            raise ValueError(
                f"Expected {expected_size} amplitudes for {self.num_qubits} qubits, "
                f"got {len(self.amplitudes)}"
            )
        if not self.qubit_ids:
            self.qubit_ids = [QubitId(f"q{i}") for i in range(self.num_qubits)]
    
    def probability(self, basis_state: int) -> float:
        """Get probability of measuring a specific basis state."""
        return float(np.abs(self.amplitudes[basis_state]) ** 2)
    
    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if state is properly normalized."""
        total_prob = float(np.sum(np.abs(self.amplitudes) ** 2))
        return abs(total_prob - 1.0) < tolerance


@dataclass
class CounterExample:
    """A counterexample demonstrating specification violation."""
    
    input_state: QuantumState
    """The input quantum state that violates the specification."""
    
    output_state: QuantumState
    """The output quantum state after program execution."""
    
    violated_condition: str
    """Description of which specification condition was violated."""
    
    trace: list[str] = field(default_factory=list)
    """Execution trace leading to the violation."""
    
    additional_info: dict[str, Any] = field(default_factory=dict)
    """Additional debugging information."""
    
    def __str__(self) -> str:
        return (
            f"CounterExample(\n"
            f"  input: {self.input_state.num_qubits} qubits,\n"
            f"  violated: {self.violated_condition}\n"
            f")"
        )


@dataclass
class VerificationResult:
    """Result of a verification attempt."""
    
    status: VerificationStatus
    """The verification status."""
    
    counterexample: Optional[CounterExample] = None
    """Counterexample if status is INVALID."""
    
    time_seconds: float = 0.0
    """Time taken for verification in seconds."""
    
    solver_stats: dict[str, Any] = field(default_factory=dict)
    """Statistics from the SMT solver."""
    
    message: str = ""
    """Human-readable message about the result."""
    
    verified_conditions: list[str] = field(default_factory=list)
    """List of verification conditions that were checked."""
    
    def is_valid(self) -> bool:
        """Check if verification succeeded."""
        return self.status == VerificationStatus.VALID
    
    def is_invalid(self) -> bool:
        """Check if verification found a counterexample."""
        return self.status == VerificationStatus.INVALID


@dataclass
class SynthesisResult:
    """Result of a specification synthesis attempt."""
    
    status: SynthesisStatus
    """The synthesis status."""
    
    specification: Optional[Any] = None  # Will be Specification type
    """The synthesized specification if successful."""
    
    candidates_tried: int = 0
    """Number of candidate specifications attempted."""
    
    refinement_iterations: int = 0
    """Number of CEGIS refinement iterations performed."""
    
    time_seconds: float = 0.0
    """Time taken for synthesis in seconds."""
    
    llm_calls: int = 0
    """Number of LLM API calls made."""
    
    message: str = ""
    """Human-readable message about the result."""
    
    def is_success(self) -> bool:
        """Check if synthesis succeeded."""
        return self.status == SynthesisStatus.SUCCESS


@dataclass
class ProgramLocation:
    """A location within a quantum program."""
    
    line: int
    column: int = 0
    label: str = ""
    
    def __str__(self) -> str:
        if self.label:
            return f"{self.label} (line {self.line})"
        return f"line {self.line}"


@dataclass
class VerificationCondition:
    """A verification condition to be checked by the SMT solver."""
    
    name: str
    """Name/identifier of this VC."""
    
    formula: str
    """SMT-LIB formula representation."""
    
    location: Optional[ProgramLocation] = None
    """Program location this VC corresponds to."""
    
    assumptions: list[str] = field(default_factory=list)
    """Assumptions made for this VC."""
    
    def __str__(self) -> str:
        return f"VC({self.name})"


# Type aliases for convenience
QubitIndex = int
Amplitude = complex
Probability = float
SMTFormula = str
