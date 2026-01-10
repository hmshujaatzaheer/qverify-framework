"""
Unit tests for QVERIFY core modules.
"""

import pytest
import numpy as np


class TestQuantumProgram:
    """Tests for QuantumProgram class."""
    
    def test_from_silq(self, sample_silq_program):
        """Test creating program from Silq source."""
        from qverify.core import QuantumProgram
        
        program = QuantumProgram.from_silq(sample_silq_program, name="test")
        assert program.name == "test"
        assert program.source == sample_silq_program
    
    def test_from_openqasm(self, sample_openqasm_program):
        """Test creating program from OpenQASM source."""
        from qverify.core import QuantumProgram
        
        program = QuantumProgram.from_openqasm(sample_openqasm_program, name="test")
        assert program.name == "test"
        assert program.num_qubits >= 2
    
    def test_parse_gates(self, sample_bell_program):
        """Test gate parsing."""
        from qverify.core import QuantumProgram
        
        program = QuantumProgram.from_silq(sample_bell_program)
        assert any("H" in str(g) for g in program.gates)
    
    def test_to_dict(self, sample_silq_program):
        """Test conversion to dictionary."""
        from qverify.core import QuantumProgram
        
        program = QuantumProgram.from_silq(sample_silq_program)
        d = program.to_dict()
        
        assert "name" in d
        assert "source" in d
        assert "num_qubits" in d


class TestSpecification:
    """Tests for Specification class."""
    
    def test_from_dict(self, sample_spec_dict):
        """Test creating specification from dictionary."""
        from qverify.core import Specification
        
        spec = Specification.from_dict(sample_spec_dict)
        assert spec.name == "test_spec"
        assert spec.is_complete()
    
    def test_trivial(self):
        """Test trivial specification."""
        from qverify.core import Specification
        
        spec = Specification.trivial()
        assert not spec.is_complete()
    
    def test_precondition_from_string(self):
        """Test precondition parsing."""
        from qverify.core import Precondition
        
        pre = Precondition.from_string("in_basis(q0, |0⟩)")
        assert "basis" in pre.to_human_readable().lower() or "q0" in pre.to_human_readable()
    
    def test_postcondition_from_string(self):
        """Test postcondition parsing."""
        from qverify.core import Postcondition
        
        post = Postcondition.from_string("superposition(q0)")
        assert "superposition" in post.to_human_readable().lower()
    
    def test_to_smt(self, sample_spec_dict):
        """Test SMT conversion."""
        from qverify.core import Specification
        
        spec = Specification.from_dict(sample_spec_dict)
        smt = spec.to_smt()
        
        assert "precondition" in smt
        assert "postcondition" in smt


class TestQuantumState:
    """Tests for QuantumState class."""
    
    def test_creation(self, sample_quantum_state):
        """Test quantum state creation."""
        assert sample_quantum_state.num_qubits == 1
        assert sample_quantum_state.is_normalized()
    
    def test_probability(self, sample_quantum_state):
        """Test probability calculation."""
        prob_0 = sample_quantum_state.probability(0)
        prob_1 = sample_quantum_state.probability(1)
        
        assert abs(prob_0 - 0.5) < 0.01
        assert abs(prob_1 - 0.5) < 0.01
        assert abs(prob_0 + prob_1 - 1.0) < 0.01
    
    def test_bell_state(self, sample_bell_state):
        """Test Bell state properties."""
        prob_00 = sample_bell_state.probability(0)  # |00⟩
        prob_11 = sample_bell_state.probability(3)  # |11⟩
        
        assert abs(prob_00 - 0.5) < 0.01
        assert abs(prob_11 - 0.5) < 0.01


class TestQuantumPredicates:
    """Tests for quantum state predicates."""
    
    def test_basis_predicate(self):
        """Test basis predicate."""
        from qverify.core.quantum_state import BasisPredicate, BasisState
        
        pred = BasisPredicate(qubit="q0", basis=BasisState.ZERO)
        assert "q0" in pred.to_human_readable()
        assert "|0" in pred.to_human_readable()
    
    def test_entanglement_predicate(self):
        """Test entanglement predicate."""
        from qverify.core.quantum_state import EntanglementPredicate
        
        pred = EntanglementPredicate(qubits=["q0", "q1"])
        assert "entangled" in pred.to_human_readable().lower()
    
    def test_superposition_predicate(self):
        """Test superposition predicate."""
        from qverify.core.quantum_state import SuperpositionPredicate
        
        pred = SuperpositionPredicate(qubit="q0")
        assert "superposition" in pred.to_human_readable().lower()
    
    def test_probability_predicate(self):
        """Test probability predicate."""
        from qverify.core.quantum_state import ProbabilityPredicate
        
        pred = ProbabilityPredicate(qubit="q0", outcome=0, comparison=">=", value=0.9)
        readable = pred.to_human_readable()
        assert "0.9" in readable
    
    def test_conjunction(self):
        """Test conjunction of predicates."""
        from qverify.core.quantum_state import (
            BasisPredicate, BasisState, SuperpositionPredicate, ConjunctionPredicate
        )
        
        p1 = BasisPredicate(qubit="q0", basis=BasisState.ZERO)
        p2 = SuperpositionPredicate(qubit="q1")
        
        conj = ConjunctionPredicate(predicates=[p1, p2])
        assert "∧" in conj.to_human_readable() or "and" in conj.to_smt()


class TestVerificationTypes:
    """Tests for verification result types."""
    
    def test_verification_result_valid(self):
        """Test valid verification result."""
        from qverify.core.types import VerificationResult, VerificationStatus
        
        result = VerificationResult(status=VerificationStatus.VALID)
        assert result.is_valid()
        assert not result.is_invalid()
    
    def test_verification_result_invalid(self):
        """Test invalid verification result."""
        from qverify.core.types import VerificationResult, VerificationStatus
        
        result = VerificationResult(status=VerificationStatus.INVALID)
        assert result.is_invalid()
        assert not result.is_valid()
    
    def test_synthesis_result_success(self):
        """Test successful synthesis result."""
        from qverify.core.types import SynthesisResult, SynthesisStatus
        
        result = SynthesisResult(status=SynthesisStatus.SUCCESS)
        assert result.is_success()
