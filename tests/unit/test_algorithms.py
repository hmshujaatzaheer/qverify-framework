"""
Unit tests for QVERIFY algorithms.
"""

import pytest


class TestQuantumSpecSynth:
    """Tests for QuantumSpecSynth algorithm."""
    
    def test_initialization(self, mock_llm):
        """Test algorithm initialization."""
        from qverify.algorithms import QuantumSpecSynth
        
        synth = QuantumSpecSynth(llm=mock_llm)
        assert synth.max_candidates == 5
        assert synth.max_refinement_iterations == 3
    
    def test_synthesize_basic(self, mock_llm, sample_silq_program):
        """Test basic synthesis."""
        from qverify.algorithms import QuantumSpecSynth
        from qverify.core import QuantumProgram
        
        synth = QuantumSpecSynth(llm=mock_llm)
        program = QuantumProgram.from_silq(sample_silq_program)
        
        result = synth.synthesize(program)
        
        # Should at least attempt synthesis
        assert result is not None
    
    def test_program_analysis(self, mock_llm, sample_bell_program):
        """Test program analysis."""
        from qverify.algorithms import QuantumSpecSynth
        from qverify.core import QuantumProgram
        
        synth = QuantumSpecSynth(llm=mock_llm)
        program = QuantumProgram.from_silq(sample_bell_program)
        
        analysis = synth._analyze_program(program)
        
        assert analysis.num_qubits >= 0
        assert isinstance(analysis.gate_sequence, list)


class TestPredicateLearning:
    """Tests for predicate learning algorithm."""
    
    def test_initialization(self):
        """Test learner initialization."""
        from qverify.algorithms import LearnQuantumPredicate
        
        learner = LearnQuantumPredicate()
        assert learner.max_candidates == 10
    
    def test_feature_extraction(self, sample_silq_program):
        """Test feature extraction."""
        from qverify.algorithms import LearnQuantumPredicate
        from qverify.core import QuantumProgram
        
        learner = LearnQuantumPredicate()
        program = QuantumProgram.from_silq(sample_silq_program)
        
        features = learner._extract_features(program, "entry")
        
        assert "num_qubits" in features
        assert "qubit_names" in features


class TestInvariantSynthesis:
    """Tests for loop invariant synthesis."""
    
    def test_initialization(self):
        """Test synthesizer initialization."""
        from qverify.algorithms import SynthesizeQuantumInvariant
        
        synth = SynthesizeQuantumInvariant()
        assert synth.max_unrolling == 5
    
    def test_loop_analysis(self):
        """Test loop analysis."""
        from qverify.algorithms import SynthesizeQuantumInvariant
        from qverify.core.quantum_program import Loop
        
        synth = SynthesizeQuantumInvariant()
        
        loop = Loop(variable="i", start=0, end=5, body=[])
        analysis = synth._analyze_loop(loop)
        
        assert analysis.loop_variable == "i"
        assert analysis.start_value == 0
        assert analysis.end_value == 5


class TestSpecRepair:
    """Tests for specification repair."""
    
    def test_initialization(self, mock_llm):
        """Test repairer initialization."""
        from qverify.algorithms import RepairSpecification
        
        repairer = RepairSpecification(llm=mock_llm)
        assert repairer.max_repair_attempts == 3
    
    def test_diagnosis(self, sample_silq_program):
        """Test failure diagnosis."""
        from qverify.algorithms import RepairSpecification, FailureDiagnosis
        from qverify.core import QuantumProgram, Specification
        from qverify.core.types import CounterExample, QuantumState
        import numpy as np
        
        repairer = RepairSpecification()
        program = QuantumProgram.from_silq(sample_silq_program)
        spec = Specification.trivial()
        
        cex = CounterExample(
            input_state=QuantumState(num_qubits=1, amplitudes=np.array([1, 0], dtype=complex)),
            output_state=QuantumState(num_qubits=1, amplitudes=np.array([0, 1], dtype=complex)),
            violated_condition="postcondition"
        )
        
        diagnosis = repairer._diagnose_failure(program, spec, cex)
        
        assert isinstance(diagnosis, FailureDiagnosis)
