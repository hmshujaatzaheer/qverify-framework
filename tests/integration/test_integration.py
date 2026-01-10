"""
Integration tests for QVERIFY.
"""

import pytest


class TestQVerifyIntegration:
    """Integration tests for the main QVerify class."""
    
    def test_verify_from_source(self, sample_silq_program):
        """Test verification from source code."""
        from qverify import QVerify
        from qverify.core.types import VerificationStatus
        
        qv = QVerify(llm="mock", backend="mock")
        result = qv.verify_from_source(sample_silq_program, language="silq")
        
        # Should complete without error
        assert result.status in [
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.UNKNOWN,
        ]
    
    def test_synthesize_and_verify(self, sample_silq_program):
        """Test synthesis followed by verification."""
        from qverify import QVerify, QuantumProgram
        
        qv = QVerify(llm="mock", backend="mock")
        program = QuantumProgram.from_silq(sample_silq_program)
        
        spec, result = qv.synthesize_and_verify(program, max_attempts=1)
        
        # Should complete
        assert result is not None
    
    def test_get_stats(self):
        """Test getting statistics."""
        from qverify import QVerify
        
        qv = QVerify(llm="mock", backend="mock")
        stats = qv.get_stats()
        
        assert "llm_available" in stats
        assert "smt_available" in stats


class TestBenchmarkIntegration:
    """Integration tests for QVerifyBench."""
    
    def test_benchmark_loading(self):
        """Test benchmark loading."""
        from qverify.benchmark import QVerifyBench
        
        bench = QVerifyBench(tier="T1")
        assert len(bench) > 0
    
    def test_benchmark_iteration(self):
        """Test iterating over benchmark."""
        from qverify.benchmark import QVerifyBench
        
        bench = QVerifyBench(tier="T1")
        
        for prog in bench:
            assert prog.id is not None
            assert prog.source is not None
            break  # Just test first
    
    def test_benchmark_statistics(self):
        """Test benchmark statistics."""
        from qverify.benchmark import QVerifyBench
        
        bench = QVerifyBench()
        stats = bench.get_statistics()
        
        assert "total_programs" in stats
        assert "by_tier" in stats


class TestEndToEnd:
    """End-to-end tests."""
    
    def test_hadamard_verification(self):
        """Test verification of Hadamard gate."""
        from qverify import QVerify, QuantumProgram
        from qverify.core import Specification, Precondition, Postcondition
        
        source = """def hadamard(q: quon) -> quon {
    q = H(q);
    return q;
}"""
        
        qv = QVerify(llm="mock", backend="mock")
        program = QuantumProgram.from_silq(source)
        
        spec = Specification(
            precondition=Precondition.from_string("in_basis(q, |0⟩)"),
            postcondition=Postcondition.from_string("superposition(q)"),
            name="hadamard_spec"
        )
        
        result = qv.verify(program, spec)
        
        # Should complete
        assert result is not None
    
    def test_bell_state_verification(self):
        """Test verification of Bell state preparation."""
        from qverify import QVerify, QuantumProgram
        from qverify.core import Specification, Precondition, Postcondition
        
        source = """def bell(q0: quon, q1: quon) -> (quon, quon) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}"""
        
        qv = QVerify(llm="mock", backend="mock")
        program = QuantumProgram.from_silq(source)
        
        spec = Specification(
            precondition=Precondition.from_string("in_basis(q0, |0⟩) and in_basis(q1, |0⟩)"),
            postcondition=Postcondition.from_string("entangled(q0, q1)"),
            name="bell_spec"
        )
        
        result = qv.verify(program, spec)
        
        assert result is not None
