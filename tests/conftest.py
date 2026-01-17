"""Pytest configuration and fixtures."""

import pytest
import numpy as np


class MockLLM:
    """Mock LLM for testing."""

    def generate(self, prompt: str, n: int = 1) -> list:
        return ['{"precondition": "true", "postcondition": "true", "invariants": []}'] * n


@pytest.fixture
def mock_llm():
    """Provide a mock LLM."""
    return MockLLM()


@pytest.fixture
def bell_state_program():
    """Provide a Bell state preparation program."""
    return """def bell_state(q0: qubit, q1: qubit) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}"""


@pytest.fixture
def simple_hadamard_program():
    """Provide a simple Hadamard program."""
    return """def hadamard(q: qubit) {
    q = H(q);
    return q;
}"""


@pytest.fixture
def zero_state():
    """Provide a |0> state."""
    from qverify.core.types import QuantumState
    return QuantumState(num_qubits=1, amplitudes=np.array([1.0, 0.0], dtype=complex))


@pytest.fixture
def plus_state():
    """Provide a |+> state."""
    from qverify.core.types import QuantumState
    return QuantumState(num_qubits=1, amplitudes=np.array([1.0, 1.0], dtype=complex) / np.sqrt(2))


@pytest.fixture
def bell_state():
    """Provide a Bell state."""
    from qverify.core.types import QuantumState
    return QuantumState(num_qubits=2, amplitudes=np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2))
