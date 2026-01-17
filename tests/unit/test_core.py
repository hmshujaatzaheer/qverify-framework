"""Unit tests for QVERIFY."""

import pytest
import numpy as np

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Specification, Precondition, Postcondition
from qverify.core.types import QuantumState, VerificationStatus, SynthesisStatus
from qverify.core.quantum_state import create_bell_state, is_entangled, state_fidelity


class TestQuantumProgram:
    """Tests for QuantumProgram class."""

    def test_from_silq_basic(self, simple_hadamard_program):
        """Test parsing basic Silq program."""
        program = QuantumProgram.from_silq(simple_hadamard_program)
        assert program.name == "hadamard"
        assert program.num_qubits >= 1

    def test_from_silq_bell_state(self, bell_state_program):
        """Test parsing Bell state program."""
        program = QuantumProgram.from_silq(bell_state_program)
        assert program.name == "bell_state"
        assert program.num_qubits >= 2

    def test_to_dict(self, simple_hadamard_program):
        """Test converting program to dictionary."""
        program = QuantumProgram.from_silq(simple_hadamard_program)
        data = program.to_dict()
        assert "source" in data
        assert "name" in data


class TestSpecification:
    """Tests for Specification classes."""

    def test_precondition_from_string(self):
        """Test parsing precondition."""
        pre = Precondition.from_string("in_basis(q0, |0>)")
        assert pre.formula == "in_basis(q0, |0>)"

    def test_postcondition_from_string(self):
        """Test parsing postcondition."""
        post = Postcondition.from_string("entangled(q0, q1)")
        assert "entangled" in post.formula

    def test_specification_from_dict(self):
        """Test creating specification from dictionary."""
        data = {"precondition": "true", "postcondition": "true", "invariants": [], "name": "test"}
        spec = Specification.from_dict(data)
        assert spec.name == "test"


class TestQuantumState:
    """Tests for quantum state utilities."""

    def test_zero_state(self):
        """Test creating zero state."""
        state = QuantumState.zero_state(1)
        assert state.num_qubits == 1
        assert state.probability(0) == 1.0

    def test_uniform_superposition(self):
        """Test creating uniform superposition."""
        state = QuantumState.uniform_superposition(2)
        assert state.num_qubits == 2
        assert abs(state.probability(0) - 0.25) < 1e-10

    def test_bell_state_creation(self):
        """Test Bell state creation."""
        bell = create_bell_state(0)
        assert len(bell) == 4
        assert abs(bell[0]) > 0

    def test_is_entangled(self, bell_state):
        """Test entanglement detection."""
        assert is_entangled(bell_state.amplitudes, 2)

    def test_state_fidelity(self, zero_state):
        """Test state fidelity calculation."""
        fidelity = state_fidelity(zero_state.amplitudes, zero_state.amplitudes)
        assert abs(fidelity - 1.0) < 1e-10


class TestQVerify:
    """Tests for main QVerify class."""

    def test_full_pipeline(self, simple_hadamard_program):
        """Test full synthesis and verification pipeline."""
        from qverify import QVerify
        qv = QVerify(llm="mock", backend="z3")
        program = QuantumProgram.from_silq(simple_hadamard_program)
        spec = qv.synthesize_specification(program)
        assert spec is not None
        result = qv.verify(program, spec)
        assert result is not None

    def test_load_program(self):
        """Test loading program."""
        from qverify import QVerify
        source = "def test(q: qubit) { q = H(q); return q; }"
        program = QVerify.load_program(source, language="silq")
        assert program.name == "test"


class TestBenchmark:
    """Tests for QVerifyBench."""

    def test_load_benchmark(self):
        """Test loading benchmark."""
        from qverify.benchmark import QVerifyBench
        bench = QVerifyBench(tier="T1")
        assert len(bench) > 0

    def test_get_statistics(self):
        """Test getting statistics."""
        from qverify.benchmark import QVerifyBench
        bench = QVerifyBench(tier="all")
        stats = bench.get_statistics()
        assert "total_programs" in stats
        assert stats["total_programs"] >= 500
