"""Core type definitions for QVERIFY."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np


class VerificationStatus(Enum):
    """Status of a verification attempt."""

    VALID = auto()
    INVALID = auto()
    UNKNOWN = auto()
    ERROR = auto()
    TIMEOUT = auto()


class SynthesisStatus(Enum):
    """Status of a specification synthesis attempt."""

    SUCCESS = auto()
    FAILED = auto()
    PARTIAL = auto()
    ERROR = auto()


@dataclass
class QuantumState:
    """Representation of a quantum state."""

    num_qubits: int
    amplitudes: np.ndarray
    basis_labels: Optional[List[str]] = None

    def __post_init__(self):
        """Validate state dimensions."""
        expected_dim = 2 ** self.num_qubits
        if len(self.amplitudes) != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} amplitudes for {self.num_qubits} qubits, "
                f"got {len(self.amplitudes)}"
            )

    @classmethod
    def zero_state(cls, num_qubits: int) -> "QuantumState":
        """Create |0...0> state."""
        amplitudes = np.zeros(2**num_qubits, dtype=complex)
        amplitudes[0] = 1.0
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)

    @classmethod
    def uniform_superposition(cls, num_qubits: int) -> "QuantumState":
        """Create uniform superposition state."""
        dim = 2**num_qubits
        amplitudes = np.ones(dim, dtype=complex) / np.sqrt(dim)
        return cls(num_qubits=num_qubits, amplitudes=amplitudes)

    def probability(self, outcome: int) -> float:
        """Get probability of measuring a specific outcome."""
        return abs(self.amplitudes[outcome]) ** 2

    def is_normalized(self, tolerance: float = 1e-10) -> bool:
        """Check if state is normalized."""
        norm = np.sum(np.abs(self.amplitudes) ** 2)
        return abs(norm - 1.0) < tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_qubits": self.num_qubits,
            "amplitudes_real": self.amplitudes.real.tolist(),
            "amplitudes_imag": self.amplitudes.imag.tolist(),
            "basis_labels": self.basis_labels,
        }


@dataclass
class CounterExample:
    """A counterexample demonstrating specification violation."""

    input_state: QuantumState
    output_state: QuantumState
    violated_condition: str
    trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_state": self.input_state.to_dict(),
            "output_state": self.output_state.to_dict(),
            "violated_condition": self.violated_condition,
            "trace": self.trace,
        }

    def __str__(self) -> str:
        """Return human-readable representation."""
        return (
            f"CounterExample(\n"
            f"  input: {self.input_state.num_qubits} qubits\n"
            f"  violated: {self.violated_condition}\n"
            f")"
        )


@dataclass
class VerificationCondition:
    """A verification condition to be checked by SMT solver."""

    name: str
    formula: str
    assumptions: List[str] = field(default_factory=list)

    def to_smt2(self) -> str:
        """Convert to SMT-LIB2 format."""
        lines = []
        for assumption in self.assumptions:
            lines.append(f"(assert {assumption})")
        lines.append(f"(assert (not {self.formula}))")
        lines.append("(check-sat)")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Result of a verification attempt."""

    status: VerificationStatus
    counterexample: Optional[CounterExample] = None
    time_seconds: float = 0.0
    solver_stats: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    verified_conditions: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if verification succeeded."""
        return self.status == VerificationStatus.VALID

    def is_invalid(self) -> bool:
        """Check if counterexample was found."""
        return self.status == VerificationStatus.INVALID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.name,
            "counterexample": self.counterexample.to_dict() if self.counterexample else None,
            "time_seconds": self.time_seconds,
            "solver_stats": self.solver_stats,
            "message": self.message,
            "verified_conditions": self.verified_conditions,
        }


@dataclass
class SynthesisResult:
    """Result of a specification synthesis attempt."""

    status: SynthesisStatus
    specification: Optional[Any] = None
    candidates_tried: int = 0
    refinement_iterations: int = 0
    time_seconds: float = 0.0
    llm_calls: int = 0
    message: str = ""

    def is_success(self) -> bool:
        """Check if synthesis succeeded."""
        return self.status == SynthesisStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.name,
            "specification": str(self.specification) if self.specification else None,
            "candidates_tried": self.candidates_tried,
            "refinement_iterations": self.refinement_iterations,
            "time_seconds": self.time_seconds,
            "llm_calls": self.llm_calls,
            "message": self.message,
        }


@dataclass
class Gate:
    """Representation of a quantum gate."""

    name: str
    qubits: List[str]
    params: List[float] = field(default_factory=list)
    line_number: int = 0

    def __str__(self) -> str:
        """Return string representation."""
        params_str = f"({', '.join(map(str, self.params))})" if self.params else ""
        return f"{self.name}{params_str}({', '.join(self.qubits)})"


@dataclass
class BenchmarkResult:
    """Result of running a benchmark evaluation."""

    synthesis_rate: float
    verification_rate: float
    avg_time: float
    tier_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model: str = ""
    total_programs: int = 0
    successful_syntheses: int = 0
    successful_verifications: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "synthesis_rate": self.synthesis_rate,
            "verification_rate": self.verification_rate,
            "avg_time": self.avg_time,
            "tier_results": self.tier_results,
            "model": self.model,
            "total_programs": self.total_programs,
            "successful_syntheses": self.successful_syntheses,
            "successful_verifications": self.successful_verifications,
        }
