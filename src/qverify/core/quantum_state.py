"""Quantum state representation and operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from qverify.core.types import QuantumState


ZERO = np.array([1, 0], dtype=complex)
ONE = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)


def tensor_product(*states: np.ndarray) -> np.ndarray:
    """Compute tensor product of multiple states."""
    result = states[0]
    for state in states[1:]:
        result = np.kron(result, state)
    return result


def create_bell_state(which: int = 0) -> np.ndarray:
    """Create one of the four Bell states."""
    if which == 0:
        return np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    elif which == 1:
        return np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    elif which == 2:
        return np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    else:
        return np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)


def create_ghz_state(num_qubits: int) -> np.ndarray:
    """Create GHZ state: (|0...0> + |1...1>)/sqrt(2)."""
    dim = 2 ** num_qubits
    state = np.zeros(dim, dtype=complex)
    state[0] = 1 / np.sqrt(2)
    state[-1] = 1 / np.sqrt(2)
    return state


def is_entangled(state: np.ndarray, num_qubits: int) -> bool:
    """Check if a state is entangled (not separable)."""
    if num_qubits < 2:
        return False
    if num_qubits == 2:
        matrix = state.reshape(2, 2)
        _, s, _ = np.linalg.svd(matrix)
        return np.sum(s > 1e-10) > 1
    dim1 = 2
    dim2 = 2 ** (num_qubits - 1)
    matrix = state.reshape(dim1, dim2)
    _, s, _ = np.linalg.svd(matrix)
    return np.sum(s > 1e-10) > 1


def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute fidelity between two pure states."""
    inner_product = np.vdot(state1, state2)
    return abs(inner_product) ** 2


HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=complex)


@dataclass
class QuantumStateAnalyzer:
    """Analyzer for quantum states."""

    tolerance: float = 1e-10

    def is_basis_state(self, state: QuantumState, basis: str) -> bool:
        """Check if state is in a specific basis state."""
        if state.num_qubits != 1:
            return False
        if basis in ('zero', '|0>', '0'):
            target = ZERO
        elif basis in ('one', '|1>', '1'):
            target = ONE
        elif basis in ('plus', '|+>', '+'):
            target = PLUS
        elif basis in ('minus', '|->', '-'):
            target = MINUS
        else:
            return False
        return state_fidelity(state.amplitudes, target) > 1 - self.tolerance

    def is_superposition(self, state: QuantumState) -> bool:
        """Check if state is in a non-trivial superposition."""
        non_zero_count = np.sum(np.abs(state.amplitudes) > self.tolerance)
        return non_zero_count > 1

    def is_entangled(self, state: QuantumState) -> bool:
        """Check if state is entangled."""
        return is_entangled(state.amplitudes, state.num_qubits)

    def get_measurement_probabilities(self, state: QuantumState) -> Dict[str, float]:
        """Get measurement probabilities for all basis states."""
        probs = {}
        for i, amp in enumerate(state.amplitudes):
            label = format(i, f'0{state.num_qubits}b')
            prob = abs(amp) ** 2
            if prob > self.tolerance:
                probs[label] = prob
        return probs

    def infer_predicates(self, state: QuantumState) -> List[str]:
        """Infer relevant predicates that hold for a state."""
        predicates = []
        if self.is_superposition(state):
            predicates.append("superposition(state)")
        if state.num_qubits >= 2 and self.is_entangled(state):
            predicates.append("entangled(state)")
        probs = self.get_measurement_probabilities(state)
        if len(probs) > 1:
            prob_values = list(probs.values())
            if all(abs(p - prob_values[0]) < self.tolerance for p in prob_values):
                predicates.append("uniform_superposition(state)")
        if state.num_qubits == 2:
            for i in range(4):
                bell = create_bell_state(i)
                if state_fidelity(state.amplitudes, bell) > 1 - self.tolerance:
                    predicates.append(f"bell_state(state, {i})")
                    break
        return predicates
