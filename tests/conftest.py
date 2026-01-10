"""
Pytest configuration and fixtures for QVERIFY tests.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_silq_program():
    """Sample Silq program for testing."""
    return """def hadamard_test(q: quon) -> quon {
    q = H(q);
    return q;
}"""


@pytest.fixture
def sample_bell_program():
    """Sample Bell state program."""
    return """def bell_state(q0: quon, q1: quon) -> (quon, quon) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}"""


@pytest.fixture
def sample_openqasm_program():
    """Sample OpenQASM program."""
    return """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


@pytest.fixture
def mock_llm():
    """Mock LLM interface for testing."""
    from qverify.utils.llm_interface import MockLLMInterface
    return MockLLMInterface()


@pytest.fixture
def sample_quantum_state():
    """Sample quantum state for testing."""
    from qverify.core.types import QuantumState
    
    # |+⟩ state
    amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    return QuantumState(num_qubits=1, amplitudes=amplitudes)


@pytest.fixture
def sample_bell_state():
    """Sample Bell state |Φ+⟩."""
    from qverify.core.types import QuantumState
    
    # |00⟩ + |11⟩ / sqrt(2)
    amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    return QuantumState(num_qubits=2, amplitudes=amplitudes)


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_spec_dict():
    """Sample specification dictionary."""
    return {
        "name": "test_spec",
        "description": "Test specification",
        "precondition": "in_basis(q0, |0⟩)",
        "postcondition": "superposition(q0)",
        "invariants": [],
        "metadata": {}
    }
