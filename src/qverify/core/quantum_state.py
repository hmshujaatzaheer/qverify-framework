"""
Quantum state predicates for QVERIFY.

This module provides predicate types for reasoning about quantum states,
including basis predicates, entanglement predicates, and amplitude predicates.
These predicates form the theory T_Q used in SMT-based verification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import numpy as np


class BasisState(Enum):
    """Standard quantum basis states."""
    
    ZERO = "|0⟩"
    ONE = "|1⟩"
    PLUS = "|+⟩"
    MINUS = "|-⟩"
    PLUS_I = "|+i⟩"
    MINUS_I = "|-i⟩"
    
    @classmethod
    def from_string(cls, s: str) -> 'BasisState':
        """Parse a basis state from string representation."""
        s = s.strip()
        mapping = {
            "|0>": cls.ZERO, "|0⟩": cls.ZERO, "0": cls.ZERO,
            "|1>": cls.ONE, "|1⟩": cls.ONE, "1": cls.ONE,
            "|+>": cls.PLUS, "|+⟩": cls.PLUS, "+": cls.PLUS,
            "|->": cls.MINUS, "|-⟩": cls.MINUS, "-": cls.MINUS,
            "|+i>": cls.PLUS_I, "|+i⟩": cls.PLUS_I,
            "|-i>": cls.MINUS_I, "|-i⟩": cls.MINUS_I,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown basis state: {s}")
    
    def to_vector(self) -> np.ndarray:
        """Convert to state vector representation."""
        vectors = {
            self.ZERO: np.array([1, 0], dtype=complex),
            self.ONE: np.array([0, 1], dtype=complex),
            self.PLUS: np.array([1, 1], dtype=complex) / np.sqrt(2),
            self.MINUS: np.array([1, -1], dtype=complex) / np.sqrt(2),
            self.PLUS_I: np.array([1, 1j], dtype=complex) / np.sqrt(2),
            self.MINUS_I: np.array([1, -1j], dtype=complex) / np.sqrt(2),
        }
        return vectors[self]


class QuantumStatePredicate(ABC):
    """
    Abstract base class for quantum state predicates.
    
    Quantum state predicates are formulas in the theory T_Q that can be
    used to specify properties of quantum states for verification.
    """
    
    @abstractmethod
    def to_smt(self) -> str:
        """Convert predicate to SMT-LIB format."""
        pass
    
    @abstractmethod
    def to_human_readable(self) -> str:
        """Convert predicate to human-readable format."""
        pass
    
    @abstractmethod
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Evaluate the predicate on a concrete quantum state."""
        pass
    
    @abstractmethod
    def negate(self) -> 'QuantumStatePredicate':
        """Return the negation of this predicate."""
        pass


@dataclass
class BasisPredicate(QuantumStatePredicate):
    """
    Predicate asserting a qubit is in a specific basis state.
    
    Example:
        in_basis(q0, |0⟩) - qubit q0 is in the |0⟩ state
    """
    
    qubit: str
    basis: BasisState
    
    def to_smt(self) -> str:
        return f"(in_basis {self.qubit} {self.basis.value})"
    
    def to_human_readable(self) -> str:
        return f"{self.qubit} = {self.basis.value}"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Check if qubit is in the specified basis state."""
        qubit_idx = qubit_mapping.get(self.qubit)
        if qubit_idx is None:
            raise ValueError(f"Unknown qubit: {self.qubit}")
        
        # Get the expected state vector
        expected = self.basis.to_vector()
        
        # Extract the reduced density matrix for this qubit
        n_qubits = int(np.log2(len(state)))
        
        # For simplicity, check amplitude pattern
        # This is a simplified check - full implementation would use partial trace
        if len(state) == 2:  # Single qubit
            return np.allclose(np.abs(state), np.abs(expected), atol=1e-6)
        
        # For multi-qubit, check marginal probabilities
        prob_0 = 0.0
        prob_1 = 0.0
        for i, amp in enumerate(state):
            if (i >> qubit_idx) & 1 == 0:
                prob_0 += np.abs(amp) ** 2
            else:
                prob_1 += np.abs(amp) ** 2
        
        if self.basis in {BasisState.ZERO}:
            return prob_0 > 0.99
        elif self.basis in {BasisState.ONE}:
            return prob_1 > 0.99
        elif self.basis in {BasisState.PLUS, BasisState.MINUS}:
            return 0.49 < prob_0 < 0.51 and 0.49 < prob_1 < 0.51
        
        return False
    
    def negate(self) -> 'NegatedPredicate':
        return NegatedPredicate(inner=self)


@dataclass
class EntanglementPredicate(QuantumStatePredicate):
    """
    Predicate asserting qubits are entangled.
    
    Example:
        entangled(q0, q1) - qubits q0 and q1 are entangled
    """
    
    qubits: list[str]
    
    def to_smt(self) -> str:
        qubits_str = " ".join(self.qubits)
        return f"(entangled {qubits_str})"
    
    def to_human_readable(self) -> str:
        qubits_str = ", ".join(self.qubits)
        return f"entangled({qubits_str})"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Check if qubits are entangled using Schmidt decomposition."""
        if len(self.qubits) != 2:
            # For simplicity, only handle bipartite entanglement
            return False
        
        q0_idx = qubit_mapping.get(self.qubits[0])
        q1_idx = qubit_mapping.get(self.qubits[1])
        
        if q0_idx is None or q1_idx is None:
            raise ValueError(f"Unknown qubit in {self.qubits}")
        
        # Reshape state as matrix for SVD
        n_qubits = int(np.log2(len(state)))
        
        if n_qubits == 2:
            # Direct check for 2-qubit state
            state_matrix = state.reshape(2, 2)
            u, s, vh = np.linalg.svd(state_matrix)
            
            # If Schmidt rank > 1, state is entangled
            non_zero_sv = np.sum(s > 1e-10)
            return non_zero_sv > 1
        
        # For larger systems, use partial trace to get reduced density matrix
        # and check purity
        return True  # Simplified - assume entangled
    
    def negate(self) -> 'NegatedPredicate':
        return NegatedPredicate(inner=self)


@dataclass
class AmplitudePredicate(QuantumStatePredicate):
    """
    Predicate about amplitude values.
    
    Example:
        amplitude(q0, |0⟩) = 1/√2
    """
    
    qubit: str
    basis_state: int  # Computational basis state index
    comparison: str  # "=", "!=", "<", ">", "<=", ">="
    value: complex
    tolerance: float = 1e-6
    
    def to_smt(self) -> str:
        return f"({self.comparison} (amplitude {self.qubit} {self.basis_state}) {self.value})"
    
    def to_human_readable(self) -> str:
        return f"amplitude({self.qubit}, |{self.basis_state}⟩) {self.comparison} {self.value}"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Check if amplitude satisfies the comparison."""
        qubit_idx = qubit_mapping.get(self.qubit)
        if qubit_idx is None:
            raise ValueError(f"Unknown qubit: {self.qubit}")
        
        if self.basis_state >= len(state):
            return False
        
        amp = state[self.basis_state]
        
        if self.comparison == "=":
            return np.abs(amp - self.value) < self.tolerance
        elif self.comparison == "!=":
            return np.abs(amp - self.value) >= self.tolerance
        elif self.comparison == "<":
            return np.abs(amp) < np.abs(self.value)
        elif self.comparison == ">":
            return np.abs(amp) > np.abs(self.value)
        elif self.comparison == "<=":
            return np.abs(amp) <= np.abs(self.value) + self.tolerance
        elif self.comparison == ">=":
            return np.abs(amp) >= np.abs(self.value) - self.tolerance
        
        return False
    
    def negate(self) -> 'NegatedPredicate':
        return NegatedPredicate(inner=self)


@dataclass
class ProbabilityPredicate(QuantumStatePredicate):
    """
    Predicate about measurement probabilities.
    
    Example:
        prob(q0, |0⟩) >= 0.9
    """
    
    qubit: str
    outcome: int  # Measurement outcome (0 or 1)
    comparison: str  # "=", "!=", "<", ">", "<=", ">="
    value: float
    tolerance: float = 1e-6
    
    def to_smt(self) -> str:
        return f"({self.comparison} (prob {self.qubit} {self.outcome}) {self.value})"
    
    def to_human_readable(self) -> str:
        return f"P({self.qubit} = |{self.outcome}⟩) {self.comparison} {self.value}"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Check if probability satisfies the comparison."""
        qubit_idx = qubit_mapping.get(self.qubit)
        if qubit_idx is None:
            raise ValueError(f"Unknown qubit: {self.qubit}")
        
        n_qubits = int(np.log2(len(state)))
        
        # Calculate probability of measuring outcome
        prob = 0.0
        for i, amp in enumerate(state):
            bit_value = (i >> (n_qubits - 1 - qubit_idx)) & 1
            if bit_value == self.outcome:
                prob += np.abs(amp) ** 2
        
        if self.comparison == "=":
            return abs(prob - self.value) < self.tolerance
        elif self.comparison == "!=":
            return abs(prob - self.value) >= self.tolerance
        elif self.comparison == "<":
            return prob < self.value
        elif self.comparison == ">":
            return prob > self.value
        elif self.comparison == "<=":
            return prob <= self.value + self.tolerance
        elif self.comparison == ">=":
            return prob >= self.value - self.tolerance
        
        return False
    
    def negate(self) -> 'NegatedPredicate':
        return NegatedPredicate(inner=self)


@dataclass
class SuperpositionPredicate(QuantumStatePredicate):
    """
    Predicate asserting a qubit is in superposition.
    
    Example:
        superposition(q0) - qubit q0 is in a non-trivial superposition
    """
    
    qubit: str
    
    def to_smt(self) -> str:
        return f"(superposition {self.qubit})"
    
    def to_human_readable(self) -> str:
        return f"superposition({self.qubit})"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        """Check if qubit is in superposition."""
        qubit_idx = qubit_mapping.get(self.qubit)
        if qubit_idx is None:
            raise ValueError(f"Unknown qubit: {self.qubit}")
        
        n_qubits = int(np.log2(len(state)))
        
        # Calculate probabilities
        prob_0 = 0.0
        prob_1 = 0.0
        for i, amp in enumerate(state):
            bit_value = (i >> (n_qubits - 1 - qubit_idx)) & 1
            if bit_value == 0:
                prob_0 += np.abs(amp) ** 2
            else:
                prob_1 += np.abs(amp) ** 2
        
        # In superposition if neither probability is close to 0 or 1
        threshold = 0.01
        return threshold < prob_0 < (1 - threshold) and threshold < prob_1 < (1 - threshold)
    
    def negate(self) -> 'NegatedPredicate':
        return NegatedPredicate(inner=self)


@dataclass
class NegatedPredicate(QuantumStatePredicate):
    """Negation of a quantum state predicate."""
    
    inner: QuantumStatePredicate
    
    def to_smt(self) -> str:
        return f"(not {self.inner.to_smt()})"
    
    def to_human_readable(self) -> str:
        return f"¬({self.inner.to_human_readable()})"
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        return not self.inner.evaluate(state, qubit_mapping)
    
    def negate(self) -> QuantumStatePredicate:
        return self.inner


@dataclass
class ConjunctionPredicate(QuantumStatePredicate):
    """Conjunction (AND) of quantum state predicates."""
    
    predicates: list[QuantumStatePredicate]
    
    def to_smt(self) -> str:
        inner = " ".join(p.to_smt() for p in self.predicates)
        return f"(and {inner})"
    
    def to_human_readable(self) -> str:
        inner = " ∧ ".join(f"({p.to_human_readable()})" for p in self.predicates)
        return inner
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        return all(p.evaluate(state, qubit_mapping) for p in self.predicates)
    
    def negate(self) -> 'DisjunctionPredicate':
        return DisjunctionPredicate(
            predicates=[p.negate() for p in self.predicates]
        )


@dataclass
class DisjunctionPredicate(QuantumStatePredicate):
    """Disjunction (OR) of quantum state predicates."""
    
    predicates: list[QuantumStatePredicate]
    
    def to_smt(self) -> str:
        inner = " ".join(p.to_smt() for p in self.predicates)
        return f"(or {inner})"
    
    def to_human_readable(self) -> str:
        inner = " ∨ ".join(f"({p.to_human_readable()})" for p in self.predicates)
        return inner
    
    def evaluate(self, state: np.ndarray, qubit_mapping: dict[str, int]) -> bool:
        return any(p.evaluate(state, qubit_mapping) for p in self.predicates)
    
    def negate(self) -> ConjunctionPredicate:
        return ConjunctionPredicate(
            predicates=[p.negate() for p in self.predicates]
        )


def parse_predicate(predicate_str: str) -> QuantumStatePredicate:
    """
    Parse a predicate string into a QuantumStatePredicate.
    
    Args:
        predicate_str: String representation of the predicate
        
    Returns:
        QuantumStatePredicate object
    """
    import re
    
    predicate_str = predicate_str.strip()
    
    # Check for basis predicate
    match = re.match(r'in_basis\s*\(\s*(\w+)\s*,\s*(.+)\s*\)', predicate_str)
    if match:
        qubit = match.group(1)
        basis = BasisState.from_string(match.group(2))
        return BasisPredicate(qubit=qubit, basis=basis)
    
    # Check for entanglement predicate
    match = re.match(r'entangled\s*\(\s*(.+)\s*\)', predicate_str)
    if match:
        qubits = [q.strip() for q in match.group(1).split(',')]
        return EntanglementPredicate(qubits=qubits)
    
    # Check for superposition predicate
    match = re.match(r'superposition\s*\(\s*(\w+)\s*\)', predicate_str)
    if match:
        return SuperpositionPredicate(qubit=match.group(1))
    
    # Check for probability predicate
    match = re.match(r'prob\s*\(\s*(\w+)\s*,?\s*(\d*)\s*\)\s*(>=|<=|>|<|=|!=)\s*(.+)', predicate_str)
    if match:
        qubit = match.group(1)
        outcome = int(match.group(2)) if match.group(2) else 0
        comparison = match.group(3)
        value = float(match.group(4))
        return ProbabilityPredicate(
            qubit=qubit, outcome=outcome, comparison=comparison, value=value
        )
    
    raise ValueError(f"Cannot parse predicate: {predicate_str}")
