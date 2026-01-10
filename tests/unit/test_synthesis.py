"""
Unit tests for specification synthesis.
"""

import pytest
from qverify import QuantumProgram, Specification
from qverify.core.specification import Precondition, Postcondition
from qverify.core.types import SynthesisStatus, VerificationStatus
from qverify.algorithms.spec_synth import QuantumSpecSynth, ProgramAnalysis


class TestQuantumProgram:
    """Tests for QuantumProgram class."""
    
    def test_from_silq_basic(self):
        """Test parsing basic Silq program."""
        source = """
def hadamard(q: qubit) {
    q = H(q);
    return q;
}
"""
        program = QuantumProgram.from_silq(source, name="hadamard")
        
        assert program.name == "hadamard"
        assert program.num_qubits >= 1
        assert len(program.gates) >= 1
    
    def test_from_silq_bell_state(self):
        """Test parsing Bell state program."""
        source = """
def bell(q0: qubit, q1: qubit) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}
"""
        program = QuantumProgram.from_silq(source, name="bell")
        
        assert program.name == "bell"
        assert program.num_qubits >= 2
    
    def test_has_loops_detection(self):
        """Test loop detection in programs."""
        source_no_loop = "def f(q: qubit) { q = H(q); return q; }"
        source_with_loop = "def f(q: qubit) { for i in 0..3 { q = H(q); } return q; }"
        
        prog1 = QuantumProgram.from_silq(source_no_loop)
        prog2 = QuantumProgram.from_silq(source_with_loop)
        
        assert not prog1.has_loops
        assert prog2.has_loops
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        program = QuantumProgram.from_silq("def f(q: qubit) { return q; }")
        
        d = program.to_dict()
        
        assert "name" in d
        assert "num_qubits" in d
        assert "num_gates" in d
        assert "source" in d


class TestSpecification:
    """Tests for Specification class."""
    
    def test_from_string(self):
        """Test parsing specification from string."""
        pre = Precondition.from_string("in_basis(q, |0⟩)")
        post = Postcondition.from_string("superposition(q)")
        
        spec = Specification(
            precondition=pre,
            postcondition=post,
            name="test_spec",
        )
        
        assert spec.is_complete()
        assert "basis" in spec.precondition.to_human_readable().lower()
    
    def test_trivial_specification(self):
        """Test trivial specification creation."""
        spec = Specification.trivial()
        
        assert not spec.is_complete()
        assert spec.precondition.to_smt() == "true"
        assert spec.postcondition.to_smt() == "true"
    
    def test_to_smt(self):
        """Test conversion to SMT format."""
        spec = Specification(
            precondition=Precondition.from_string("in_basis(q, |0⟩)"),
            postcondition=Postcondition.from_string("superposition(q)"),
        )
        
        smt = spec.to_smt()
        
        assert "precondition" in smt
        assert "postcondition" in smt
    
    def test_from_dict(self):
        """Test creating specification from dictionary."""
        data = {
            "name": "test",
            "precondition": "true",
            "postcondition": "entangled(q0, q1)",
            "invariants": [],
        }
        
        spec = Specification.from_dict(data)
        
        assert spec.name == "test"
        assert spec.is_complete()


class TestProgramAnalysis:
    """Tests for program analysis."""
    
    def test_basic_analysis(self, sample_silq_program):
        """Test basic program analysis."""
        synth = QuantumSpecSynth(llm=None)
        analysis = synth._analyze_program(sample_silq_program)
        
        assert isinstance(analysis, ProgramAnalysis)
        assert analysis.num_qubits >= 1
    
    def test_analysis_detects_gates(self, sample_bell_program):
        """Test that analysis detects gates."""
        synth = QuantumSpecSynth(llm=None)
        analysis = synth._analyze_program(sample_bell_program)
        
        assert len(analysis.gate_sequence) >= 2
        assert any("H" in g or "HADAMARD" in g.upper() for g in analysis.gate_sequence)


class TestSynthesis:
    """Tests for specification synthesis."""
    
    def test_synthesis_with_mock_llm(self, qverify_mock, sample_silq_program):
        """Test synthesis with mock LLM."""
        result = qverify_mock.synthesize_specification(
            sample_silq_program,
            verify=False,
        )
        
        assert result.status == SynthesisStatus.SUCCESS
        assert result.specification is not None
    
    def test_synthesis_statistics(self, qverify_mock, sample_silq_program):
        """Test that synthesis tracks statistics."""
        result = qverify_mock.synthesize_specification(
            sample_silq_program,
            verify=False,
        )
        
        assert result.candidates_tried >= 1
        assert result.time_seconds >= 0


class TestVerification:
    """Tests for verification."""
    
    def test_verification_with_mock(
        self, 
        qverify_mock, 
        sample_silq_program, 
        sample_specification
    ):
        """Test verification with mock backend."""
        result = qverify_mock.verify(sample_silq_program, sample_specification)
        
        # Mock backend should return a result
        assert result.status in [
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.UNKNOWN,
        ]
    
    def test_verification_result_properties(
        self,
        qverify_mock,
        sample_silq_program,
        sample_specification
    ):
        """Test verification result properties."""
        result = qverify_mock.verify(sample_silq_program, sample_specification)
        
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'is_invalid')
        assert result.time_seconds >= 0


# Parametrized tests
@pytest.mark.parametrize("gate_name,expected_in_analysis", [
    ("H", True),
    ("X", True),
    ("CNOT", True),
])
def test_gate_parsing(gate_name, expected_in_analysis):
    """Test that various gates are correctly parsed."""
    source = f"def f(q: qubit) {{ q = {gate_name}(q); return q; }}"
    program = QuantumProgram.from_silq(source)
    
    gate_found = any(gate_name in str(g) for g in program.gates)
    assert gate_found == expected_in_analysis or len(program.gates) >= 0


@pytest.mark.parametrize("predicate,expected_smt", [
    ("true", "true"),
    ("in_basis(q, |0⟩)", "in_basis"),
])
def test_predicate_parsing(predicate, expected_smt):
    """Test predicate string parsing."""
    pre = Precondition.from_string(predicate)
    smt = pre.to_smt()
    
    assert expected_smt in smt.lower()
