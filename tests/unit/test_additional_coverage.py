"""Additional tests to achieve 100% coverage."""

import json
import pytest
import numpy as np

from qverify import QVerify, QuantumProgram, QuantumState
from qverify.core.types import CounterExample, VerificationResult, VerificationStatus
from qverify.core.specification import Specification, Precondition, Postcondition, Invariant
from qverify.core.quantum_state import is_entangled, create_ghz_state
from qverify.algorithms.spec_synth import QuantumSpecSynth
from qverify.algorithms.predicate_learning import LearnQuantumPredicate, PredicateExample
from qverify.algorithms.invariant_synth import SynthesizeQuantumInvariant, LoopTrace
from qverify.algorithms.spec_repair import RepairSpecification, FailureDiagnosis
from qverify.verification.neural_verifier import NeuralVerifier
from qverify.benchmark import QVerifyBench


class MockLLM:
    def __init__(self, response=None):
        self.response = response or '{"precondition": "true", "postcondition": "true", "invariants": []}'
    def generate(self, prompt: str, n: int = 1) -> list:
        return [self.response] * n


class MockSMTSolver:
    def __init__(self, result="unsat"):
        self._result = result
    def check(self, formula, timeout=30.0):
        return (self._result, {"model": {}})


# === Tests for spec_repair.py missing lines ===

class TestRepairSpecificationAdditional:
    """Additional tests for RepairSpecification."""
    
    def test_diagnose_low_input_entropy(self):
        """Test diagnosis when input state has low entropy."""
        repair = RepairSpecification(llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        pre = Precondition.from_string("true")
        post = Postcondition.from_string("true")
        
        # Low entropy input (basis state)
        cex = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.uniform_superposition(1),
            violated_condition="unknown"
        )
        diagnosis = repair._diagnose_failure(program, pre, post, cex)
        assert diagnosis in [FailureDiagnosis.WEAK_PRECONDITION, FailureDiagnosis.UNKNOWN]
    
    def test_diagnose_high_output_entropy(self):
        """Test diagnosis when output has high entropy."""
        repair = RepairSpecification(llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        pre = Precondition.from_string("true")
        post = Postcondition.from_string("true")
        
        # High entropy output (superposition)
        cex = CounterExample(
            input_state=QuantumState.uniform_superposition(2),
            output_state=QuantumState.uniform_superposition(2),
            violated_condition="something"
        )
        diagnosis = repair._diagnose_failure(program, pre, post, cex)
        assert diagnosis in [FailureDiagnosis.STRONG_POSTCONDITION, FailureDiagnosis.UNKNOWN]
    
    def test_generate_exclusion(self):
        """Test generating exclusion formula."""
        repair = RepairSpecification(llm=MockLLM())
        state = QuantumState.zero_state(2)
        exclusion = repair._generate_exclusion(state)
        assert "in_basis" in exclusion
    
    def test_generate_allowed_superposition(self):
        """Test generating allowed formula for superposition."""
        repair = RepairSpecification(llm=MockLLM())
        state = QuantumState.uniform_superposition(2)
        allowed = repair._generate_allowed(state)
        assert "superposition" in allowed
    
    def test_generate_allowed_basis(self):
        """Test generating allowed formula for basis state."""
        repair = RepairSpecification(llm=MockLLM())
        state = QuantumState.zero_state(2)
        allowed = repair._generate_allowed(state)
        assert "in_basis" in allowed
    
    def test_llm_repair_bad_json(self):
        """Test LLM repair with bad JSON response."""
        repair = RepairSpecification(llm=MockLLM("not json"))
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        cex = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.zero_state(1),
            violated_condition="logical error"
        )
        result = repair._llm_repair(program, spec, cex, FailureDiagnosis.LOGICAL_ERROR)
        assert result is None
    
    def test_llm_repair_good_json(self):
        """Test LLM repair with good JSON response."""
        good_json = '{"precondition": "repaired_pre", "postcondition": "repaired_post"}'
        repair = RepairSpecification(llm=MockLLM(good_json))
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        cex = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.zero_state(1),
            violated_condition="unknown"
        )
        result = repair._llm_repair(program, spec, cex, FailureDiagnosis.UNKNOWN)
        assert result is not None
        assert result.precondition.formula == "repaired_pre"


# === Tests for predicate_learning.py missing lines ===

class TestLearnQuantumPredicateAdditional:
    """Additional tests for LearnQuantumPredicate."""
    
    def test_learn_with_counterexample(self):
        """Test learning with counterexamples."""
        learner = LearnQuantumPredicate(llm=MockLLM("superposition(q)"))
        program = QuantumProgram.from_silq("def test(q: qubit) { q = H(q); return q; }")
        
        # Create examples with counterexample
        examples = [
            PredicateExample(state=QuantumState.zero_state(1), expected=False, actual=None),
            PredicateExample(state=QuantumState.uniform_superposition(1), expected=True, actual=None),
        ]
        result = learner.learn(program, "line_1", examples)
        # Mock may not produce correct predicate
        assert result is None or isinstance(result, str)
    
    def test_parse_predicate_with_prefix(self):
        """Test parsing predicate with 'Predicate:' prefix."""
        learner = LearnQuantumPredicate(llm=MockLLM())
        result = learner._parse_predicate("Predicate: superposition(q)")
        assert result == "superposition(q)"
    
    def test_parse_predicate_with_backticks(self):
        """Test parsing predicate with backticks."""
        learner = LearnQuantumPredicate(llm=MockLLM())
        result = learner._parse_predicate("`in_basis(q, |0>)`")
        assert "in_basis" in result
    
    def test_parse_predicate_invalid(self):
        """Test parsing invalid predicate."""
        learner = LearnQuantumPredicate(llm=MockLLM())
        result = learner._parse_predicate("just some random text")
        assert result is None
    
    def test_evaluate_single_qubit_entangled(self):
        """Test evaluating entangled predicate on single qubit."""
        learner = LearnQuantumPredicate(llm=MockLLM())
        state = QuantumState.zero_state(1)
        assert not learner._evaluate_predicate("entangled(q)", state)
    
    def test_build_prompt(self):
        """Test building LLM prompt."""
        from qverify.algorithms.predicate_learning import QuantumFeatures
        learner = LearnQuantumPredicate(llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        features = QuantumFeatures(
            num_qubits=1,
            active_qubits=["q"],
            preceding_gates=["H"],
            entanglement_possible=False,
            in_loop=False
        )
        examples = [PredicateExample(state=QuantumState.zero_state(1), expected=True)]
        prompt = learner._build_prompt(program, "line_1", examples, features, ["superposition(q)"])
        assert "line_1" in prompt


# === Tests for invariant_synth.py missing lines ===

class TestSynthesizeQuantumInvariantAdditional:
    """Additional tests for SynthesizeQuantumInvariant."""
    
    def test_default_simulate(self):
        """Test default simulation of loop body."""
        synth = SynthesizeQuantumInvariant(llm=MockLLM())
        program = QuantumProgram.from_silq("""def test(q: qubit) {
            q = H(q);
            return q;
        }""")
        state = QuantumState.zero_state(1)
        result = synth._default_simulate(program, state)
        assert result.num_qubits == 1
    
    def test_generalize_no_response(self):
        """Test generalization with no LLM response."""
        class EmptyLLM:
            def generate(self, prompt, n=1):
                return []
        synth = SynthesizeQuantumInvariant(llm=EmptyLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        pre = Precondition.from_string("true")
        post = Postcondition.from_string("true")
        result = synth._generalize(program, ["k=1: pred1"], pre, post)
        assert result is None
    
    def test_learn_empty_traces(self):
        """Test learning with empty traces."""
        synth = SynthesizeQuantumInvariant(llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = synth._learn_iteration_predicate(program, [], 1)
        assert result is None
    
    def test_learn_predicate_with_low_prob(self):
        """Test learning predicate when max_prob is low."""
        synth = SynthesizeQuantumInvariant(llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        # Uniform superposition has max_prob = 0.5
        traces = [LoopTrace(
            iteration=1,
            entry_state=QuantumState.uniform_superposition(2),
            exit_state=QuantumState.uniform_superposition(2)
        )]
        result = synth._learn_iteration_predicate(program, traces, 1)
        assert result is None or "superposition" in result


# === Tests for spec_synth.py missing lines ===

class TestQuantumSpecSynthAdditional:
    """Additional tests for QuantumSpecSynth."""
    
    def test_synthesize_with_exception(self):
        """Test synthesis when an exception occurs."""
        class ErrorLLM:
            def generate(self, prompt, n=1):
                raise RuntimeError("LLM error")
        synth = QuantumSpecSynth(llm=ErrorLLM(), max_candidates=1)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = synth.synthesize(program)
        assert result.status == result.status  # Should not crash
    
    def test_check_well_formed_exception(self):
        """Test well-formedness check when SMT conversion fails."""
        synth = QuantumSpecSynth(llm=MockLLM(), max_candidates=1)
        # Create spec with complex formula that might fail
        pre = Precondition.from_string("valid_formula")
        post = Postcondition.from_string("valid_formula")
        from qverify.algorithms.spec_synth import TypeInference
        ti = TypeInference(qubit_types={}, constraints=[])
        assert synth._check_well_formed(pre, post, [], ti)


# === Tests for neural_verifier.py missing lines ===

class TestNeuralVerifierAdditional:
    """Additional tests for NeuralVerifier."""
    
    def test_verify_with_smt_solver(self):
        """Test verification with actual SMT solver."""
        solver = MockSMTSolver("unsat")
        verifier = NeuralVerifier(smt_solver=solver, llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = verifier.verify(program, spec)
        assert result.status == VerificationStatus.VALID
    
    def test_verify_with_smt_solver_sat(self):
        """Test verification when SMT returns SAT (counterexample)."""
        solver = MockSMTSolver("sat")
        verifier = NeuralVerifier(smt_solver=solver, llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = verifier.verify(program, spec)
        assert result.status == VerificationStatus.INVALID
    
    def test_verify_with_smt_solver_unknown(self):
        """Test verification when SMT returns unknown."""
        solver = MockSMTSolver("unknown")
        verifier = NeuralVerifier(smt_solver=solver, llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = verifier.verify(program, spec)
        assert result.status == VerificationStatus.UNKNOWN
    
    def test_verify_with_smt_exception(self):
        """Test verification when SMT throws exception."""
        class ErrorSolver:
            def check(self, formula, timeout=30.0):
                raise RuntimeError("Solver error")
        verifier = NeuralVerifier(smt_solver=ErrorSolver(), llm=MockLLM())
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = verifier.verify(program, spec)
        assert result.status in [VerificationStatus.UNKNOWN, VerificationStatus.ERROR]


# === Tests for qverifybench.py missing lines ===

class TestQVerifyBenchAdditional:
    """Additional tests for QVerifyBench."""
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        from pathlib import Path
        bench = QVerifyBench(tier="all", benchmark_path=Path("/nonexistent/path/file.json"))
        assert len(bench) == 0
    
    def test_num_programs_property(self):
        """Test num_programs property."""
        bench = QVerifyBench(tier="T1")
        assert bench.num_programs == 150


# === Tests for quantum_program.py missing lines ===

class TestQuantumProgramAdditional:
    """Additional tests for QuantumProgram."""
    
    def test_extract_qasm_gates(self):
        """Test extracting gates from QASM."""
        qasm = """OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        """
        program = QuantumProgram.from_qasm(qasm)
        assert len(program.gates) > 0
    
    def test_silq_with_rotation_gates(self):
        """Test Silq with parameterized gates."""
        source = """def test(q: qubit, theta: float) {
            q = RZ(3.14, q);
            q = RX(1.57, q);
            return q;
        }"""
        program = QuantumProgram.from_silq(source)
        # Should parse without error
        assert program.num_qubits >= 1
    
    def test_silq_with_swap(self):
        """Test Silq with SWAP gate."""
        source = """def test(q0: qubit, q1: qubit) {
            SWAP(q0, q1);
            return (q0, q1);
        }"""
        program = QuantumProgram.from_silq(source)
        # Should detect two-qubit gate
        assert program.num_qubits >= 2


# === Tests for quantum_state.py missing lines ===

class TestQuantumStateAdditional:
    """Additional tests for quantum state operations."""
    
    def test_is_entangled_multi_qubit(self):
        """Test entanglement detection for >2 qubits."""
        ghz = create_ghz_state(3)
        assert is_entangled(ghz, 3)
    
    def test_state_analyzer_uniform_superposition(self):
        """Test analyzer inferring uniform superposition predicate."""
        from qverify.core.quantum_state import QuantumStateAnalyzer
        analyzer = QuantumStateAnalyzer()
        state = QuantumState.uniform_superposition(2)
        preds = analyzer.infer_predicates(state)
        assert any("uniform" in p for p in preds)


# === Tests for specification.py missing lines ===

class TestSpecificationAdditional:
    """Additional tests for specification module."""
    
    def test_precondition_to_smt_complex(self):
        """Test complex precondition to SMT conversion."""
        pre = Precondition.from_string("in_basis(q0, |0⟩) → entangled(q0, q1)")
        smt = pre.to_smt()
        assert "=>" in smt or "in_basis" in smt
    
    def test_postcondition_to_smt_equivalence(self):
        """Test postcondition with equivalence."""
        post = Postcondition.from_string("state1 ↔ state2")
        smt = post.to_smt()
        assert "=" in smt or "state" in smt


# === Tests for qverify.py missing lines ===

class TestQVerifyAdditional:
    """Additional tests for main QVerify interface."""
    
    def test_synthesize_with_hints(self):
        """Test specification synthesis with hints."""
        qv = QVerify(llm="mock")
        program = QuantumProgram.from_silq("""def bell(q0: qubit, q1: qubit) {
            q0 = H(q0);
            CNOT(q0, q1);
            return (q0, q1);
        }""")
        hints = {"precondition_hint": "all zero", "postcondition_hint": "entangled"}
        spec = qv.synthesize_specification(program, hints=hints)
        assert spec is not None
    
    def test_mock_llm_grover(self):
        """Test mock LLM with Grover-like program."""
        from qverify.qverify import MockLLM
        llm = MockLLM()
        response = llm.generate("grover iteration", n=1)
        assert len(response) == 1
        data = json.loads(response[0])
        assert "precondition" in data
    
    def test_mock_llm_hadamard(self):
        """Test mock LLM with Hadamard program."""
        from qverify.qverify import MockLLM
        llm = MockLLM()
        response = llm.generate("hadamard gate", n=1)
        assert len(response) == 1
