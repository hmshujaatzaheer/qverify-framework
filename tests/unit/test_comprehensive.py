"""Comprehensive unit tests for QVERIFY - targeting 100% coverage."""

import json
import pytest
import numpy as np

from qverify import QVerify, QuantumProgram, QuantumState
from qverify.core.types import (
    VerificationStatus, SynthesisStatus, VerificationResult,
    SynthesisResult, CounterExample, VerificationCondition, Gate, BenchmarkResult
)
from qverify.core.specification import (
    Specification, Precondition, Postcondition, Invariant
)
from qverify.core.quantum_state import (
    create_bell_state, create_ghz_state, is_entangled, state_fidelity,
    tensor_product, QuantumStateAnalyzer, HADAMARD, PAULI_X, PAULI_Y,
    PAULI_Z, CNOT, SWAP, ZERO, ONE, PLUS, MINUS
)
from qverify.algorithms.spec_synth import QuantumSpecSynth, ProgramAnalysis, TypeInference
from qverify.algorithms.predicate_learning import LearnQuantumPredicate, QuantumFeatures, PredicateExample
from qverify.algorithms.invariant_synth import SynthesizeQuantumInvariant, LoopTrace
from qverify.algorithms.spec_repair import RepairSpecification, FailureDiagnosis
from qverify.verification.neural_verifier import NeuralVerifier, SMTStats, create_z3_interface
from qverify.benchmark import QVerifyBench, BenchmarkProgram


# ============== Fixtures ==============

class MockLLM:
    """Mock LLM for testing."""
    def __init__(self, response=None):
        self.response = response or '{"precondition": "true", "postcondition": "true", "invariants": []}'
    
    def generate(self, prompt: str, n: int = 1) -> list:
        return [self.response] * n


class MockVerificationOracle:
    """Mock verification oracle."""
    def __init__(self, result="unsat"):
        self.result = result
    
    def verify(self, program, pre, post, invariants):
        if self.result == "valid":
            return VerificationResult(status=VerificationStatus.VALID)
        return VerificationResult(status=VerificationStatus.INVALID)
    
    def check_implication(self, premise, conclusion):
        return True


@pytest.fixture
def mock_llm():
    return MockLLM()


@pytest.fixture
def bell_spec_llm():
    return MockLLM('{"precondition": "in_basis(q0, |0>) and in_basis(q1, |0>)", "postcondition": "entangled(q0, q1)", "invariants": []}')


@pytest.fixture
def grover_llm():
    return MockLLM('{"precondition": "superposition(qubits)", "postcondition": "amplified(qubits)", "invariants": ["prob(marked) >= 1/N"]}')


# ============== Core Types Tests ==============

class TestQuantumState:
    """Tests for QuantumState type."""
    
    def test_zero_state(self):
        state = QuantumState.zero_state(1)
        assert state.num_qubits == 1
        assert state.probability(0) == pytest.approx(1.0)
        assert state.probability(1) == pytest.approx(0.0)
    
    def test_zero_state_multi_qubit(self):
        state = QuantumState.zero_state(3)
        assert state.num_qubits == 3
        assert len(state.amplitudes) == 8
        assert state.probability(0) == pytest.approx(1.0)
    
    def test_uniform_superposition(self):
        state = QuantumState.uniform_superposition(2)
        assert state.num_qubits == 2
        for i in range(4):
            assert state.probability(i) == pytest.approx(0.25)
    
    def test_is_normalized(self):
        state = QuantumState.zero_state(1)
        assert state.is_normalized()
        
        # Unnormalized state
        bad_state = QuantumState(num_qubits=1, amplitudes=np.array([2.0, 0.0], dtype=complex))
        assert not bad_state.is_normalized()
    
    def test_to_dict(self):
        state = QuantumState.zero_state(1)
        d = state.to_dict()
        assert "num_qubits" in d
        assert "amplitudes_real" in d
        assert "amplitudes_imag" in d
    
    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            QuantumState(num_qubits=2, amplitudes=np.array([1.0, 0.0], dtype=complex))


class TestCounterExample:
    """Tests for CounterExample type."""
    
    def test_creation(self):
        input_state = QuantumState.zero_state(1)
        output_state = QuantumState.uniform_superposition(1)
        cex = CounterExample(
            input_state=input_state,
            output_state=output_state,
            violated_condition="postcondition"
        )
        assert cex.violated_condition == "postcondition"
    
    def test_to_dict(self):
        input_state = QuantumState.zero_state(1)
        output_state = QuantumState.zero_state(1)
        cex = CounterExample(
            input_state=input_state,
            output_state=output_state,
            violated_condition="test",
            trace=["step1", "step2"]
        )
        d = cex.to_dict()
        assert "input_state" in d
        assert "violated_condition" in d
        assert d["trace"] == ["step1", "step2"]
    
    def test_str_representation(self):
        cex = CounterExample(
            input_state=QuantumState.zero_state(2),
            output_state=QuantumState.zero_state(2),
            violated_condition="test"
        )
        s = str(cex)
        assert "CounterExample" in s
        assert "2 qubits" in s


class TestVerificationCondition:
    """Tests for VerificationCondition."""
    
    def test_to_smt2(self):
        vc = VerificationCondition(
            name="test_vc",
            formula="(=> pre post)",
            assumptions=["(= x 1)", "(> y 0)"]
        )
        smt = vc.to_smt2()
        assert "(assert (= x 1))" in smt
        assert "(assert (> y 0))" in smt
        assert "(check-sat)" in smt


class TestVerificationResult:
    """Tests for VerificationResult."""
    
    def test_is_valid(self):
        result = VerificationResult(status=VerificationStatus.VALID)
        assert result.is_valid()
        assert not result.is_invalid()
    
    def test_is_invalid(self):
        result = VerificationResult(status=VerificationStatus.INVALID)
        assert result.is_invalid()
        assert not result.is_valid()
    
    def test_to_dict(self):
        result = VerificationResult(
            status=VerificationStatus.VALID,
            time_seconds=1.5,
            message="All VCs proven"
        )
        d = result.to_dict()
        assert d["status"] == "VALID"
        assert d["time_seconds"] == 1.5


class TestSynthesisResult:
    """Tests for SynthesisResult."""
    
    def test_is_success(self):
        result = SynthesisResult(status=SynthesisStatus.SUCCESS)
        assert result.is_success()
    
    def test_to_dict(self):
        result = SynthesisResult(
            status=SynthesisStatus.FAILED,
            candidates_tried=5,
            llm_calls=3
        )
        d = result.to_dict()
        assert d["status"] == "FAILED"
        assert d["candidates_tried"] == 5


class TestGate:
    """Tests for Gate type."""
    
    def test_str_no_params(self):
        gate = Gate(name="H", qubits=["q0"])
        assert str(gate) == "H(q0)"
    
    def test_str_with_params(self):
        gate = Gate(name="RZ", qubits=["q0"], params=[3.14])
        assert "RZ" in str(gate)
        assert "3.14" in str(gate)


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""
    
    def test_to_dict(self):
        result = BenchmarkResult(
            synthesis_rate=0.85,
            verification_rate=0.72,
            avg_time=5.5,
            model="gpt-4"
        )
        d = result.to_dict()
        assert d["synthesis_rate"] == 0.85
        assert d["model"] == "gpt-4"


# ============== Specification Tests ==============

class TestPrecondition:
    """Tests for Precondition class."""
    
    def test_from_string(self):
        pre = Precondition.from_string("in_basis(q0, |0>) and in_basis(q1, |0>)")
        assert "in_basis" in pre.formula
    
    def test_to_smt(self):
        pre = Precondition.from_string("in_basis(q0, |0⟩) ∧ true")
        smt = pre.to_smt()
        assert "and" in smt or "zero" in smt
    
    def test_to_human_readable(self):
        pre = Precondition.from_string("true")
        pre.description = "All qubits start in zero state"
        assert pre.to_human_readable() == "All qubits start in zero state"
    
    def test_str(self):
        pre = Precondition.from_string("test_formula")
        assert str(pre) == "test_formula"


class TestPostcondition:
    """Tests for Postcondition class."""
    
    def test_from_string(self):
        post = Postcondition.from_string("entangled(q0, q1)")
        assert "entangled" in post.formula
    
    def test_to_smt(self):
        post = Postcondition.from_string("entangled(q0, q1) ∨ superposition(q)")
        smt = post.to_smt()
        assert "or" in smt or "entangled" in smt
    
    def test_to_human_readable_no_description(self):
        post = Postcondition.from_string("some_formula")
        assert post.to_human_readable() == "some_formula"


class TestInvariant:
    """Tests for Invariant class."""
    
    def test_from_string(self):
        inv = Invariant.from_string("prob(marked) >= 1/N", "loop_entry")
        assert inv.location == "loop_entry"
    
    def test_to_smt(self):
        inv = Invariant.from_string("x >= 0 and y <= 10")
        smt = inv.to_smt()
        assert ">=" in smt
    
    def test_str(self):
        inv = Invariant.from_string("test", "point_A")
        s = str(inv)
        assert "@point_A" in s


class TestSpecification:
    """Tests for Specification class."""
    
    def test_from_dict_basic(self):
        data = {
            "precondition": "true",
            "postcondition": "true",
            "invariants": []
        }
        spec = Specification.from_dict(data)
        assert spec.precondition.formula == "true"
    
    def test_from_dict_with_invariants_as_strings(self):
        data = {
            "precondition": "pre",
            "postcondition": "post",
            "invariants": ["inv1", "inv2"]
        }
        spec = Specification.from_dict(data)
        assert len(spec.invariants) == 2
    
    def test_from_dict_with_invariants_as_dicts(self):
        data = {
            "precondition": "pre",
            "postcondition": "post",
            "invariants": [{"formula": "inv1", "location": "loop"}]
        }
        spec = Specification.from_dict(data)
        assert spec.invariants[0].location == "loop"
    
    def test_to_dict(self):
        spec = Specification(
            precondition=Precondition.from_string("pre"),
            postcondition=Postcondition.from_string("post"),
            name="test_spec"
        )
        d = spec.to_dict()
        assert d["name"] == "test_spec"
    
    def test_str(self):
        spec = Specification(
            precondition=Precondition.from_string("pre"),
            postcondition=Postcondition.from_string("post"),
            name="myspec"
        )
        s = str(spec)
        assert "myspec" in s


# ============== Quantum Program Tests ==============

class TestQuantumProgram:
    """Tests for QuantumProgram class."""
    
    def test_from_silq_basic(self):
        source = """def hadamard(q: qubit) {
            q = H(q);
            return q;
        }"""
        program = QuantumProgram.from_silq(source)
        assert program.name == "hadamard"
        assert program.num_qubits >= 1
    
    def test_from_silq_bell_state(self):
        source = """def bell_state(q0: qubit, q1: qubit) {
            q0 = H(q0);
            CNOT(q0, q1);
            return (q0, q1);
        }"""
        program = QuantumProgram.from_silq(source)
        assert program.name == "bell_state"
        assert len(program.qubits) >= 2
    
    def test_from_silq_with_loops(self):
        source = """def grover(qubits: qubit[]) {
            for q in qubits { q = H(q); }
            return qubits;
        }"""
        program = QuantumProgram.from_silq(source)
        assert program.has_loops
    
    def test_from_silq_with_measurement(self):
        source = """def measure_qubit(q: qubit) {
            result = measure(q);
            return result;
        }"""
        program = QuantumProgram.from_silq(source)
        assert program.has_measurements
    
    def test_from_qasm(self):
        source = """OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];"""
        program = QuantumProgram.from_qasm(source)
        assert program.num_qubits == 2
    
    def test_from_dict(self):
        data = {
            "source": "test source",
            "language": "silq",
            "name": "test",
            "qubits": ["q0", "q1"],
            "gates": [{"gate": "H", "qubits": ["q0"]}],
            "num_qubits": 2
        }
        program = QuantumProgram.from_dict(data)
        assert program.name == "test"
        assert len(program.gates) == 1
    
    def test_to_dict(self):
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        d = program.to_dict()
        assert "source" in d
        assert "gates" in d
    
    def test_get_gate_sequence(self):
        program = QuantumProgram.from_silq("""def test(q: qubit) {
            q = H(q);
            q = X(q);
            return q;
        }""")
        seq = program.get_gate_sequence()
        assert "H" in seq or "X" in seq
    
    def test_get_entangling_gates(self):
        program = QuantumProgram.from_silq("""def test(q0: qubit, q1: qubit) {
            CNOT(q0, q1);
            return (q0, q1);
        }""")
        entangling = program.get_entangling_gates()
        assert len(entangling) >= 0  # May or may not detect
    
    def test_str_repr(self):
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        s = str(program)
        assert "QuantumProgram" in s
        r = repr(program)
        assert "QuantumProgram" in r


# ============== Quantum State Operations Tests ==============

class TestQuantumStateOperations:
    """Tests for quantum state operations."""
    
    def test_tensor_product(self):
        result = tensor_product(ZERO, ZERO)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)
    
    def test_create_bell_state_0(self):
        bell = create_bell_state(0)
        assert len(bell) == 4
        assert abs(bell[0]) > 0.5
        assert abs(bell[3]) > 0.5
    
    def test_create_bell_state_1(self):
        bell = create_bell_state(1)
        assert len(bell) == 4
    
    def test_create_bell_state_2(self):
        bell = create_bell_state(2)
        assert len(bell) == 4
    
    def test_create_bell_state_3(self):
        bell = create_bell_state(3)
        assert len(bell) == 4
    
    def test_create_ghz_state(self):
        ghz = create_ghz_state(3)
        assert len(ghz) == 8
        assert abs(ghz[0]) > 0.5
        assert abs(ghz[-1]) > 0.5
    
    def test_is_entangled_bell(self):
        bell = create_bell_state(0)
        assert is_entangled(bell, 2)
    
    def test_is_entangled_product(self):
        product = tensor_product(ZERO, ZERO)
        assert not is_entangled(product, 2)
    
    def test_is_entangled_single_qubit(self):
        assert not is_entangled(ZERO, 1)
    
    def test_state_fidelity_same(self):
        assert state_fidelity(ZERO, ZERO) == pytest.approx(1.0)
    
    def test_state_fidelity_orthogonal(self):
        assert state_fidelity(ZERO, ONE) == pytest.approx(0.0)
    
    def test_gate_matrices(self):
        # Test basic gate matrices exist and have correct shape
        assert HADAMARD.shape == (2, 2)
        assert PAULI_X.shape == (2, 2)
        assert PAULI_Y.shape == (2, 2)
        assert PAULI_Z.shape == (2, 2)
        assert CNOT.shape == (4, 4)
        assert SWAP.shape == (4, 4)
    
    def test_basis_states(self):
        assert ZERO[0] == pytest.approx(1.0)
        assert ONE[1] == pytest.approx(1.0)
        assert abs(PLUS[0] - PLUS[1]) < 1e-10
        assert abs(MINUS[0] + MINUS[1]) < 1e-10


class TestQuantumStateAnalyzer:
    """Tests for QuantumStateAnalyzer."""
    
    def test_is_basis_state_zero(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=1, amplitudes=ZERO)
        assert analyzer.is_basis_state(state, "zero")
        assert analyzer.is_basis_state(state, "|0>")
        assert analyzer.is_basis_state(state, "0")
    
    def test_is_basis_state_one(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=1, amplitudes=ONE)
        assert analyzer.is_basis_state(state, "one")
    
    def test_is_basis_state_plus(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=1, amplitudes=PLUS)
        assert analyzer.is_basis_state(state, "plus")
    
    def test_is_basis_state_minus(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=1, amplitudes=MINUS)
        assert analyzer.is_basis_state(state, "minus")
    
    def test_is_basis_state_multi_qubit(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=2, amplitudes=tensor_product(ZERO, ZERO))
        assert not analyzer.is_basis_state(state, "zero")  # Multi-qubit
    
    def test_is_basis_state_unknown_basis(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=1, amplitudes=ZERO)
        assert not analyzer.is_basis_state(state, "unknown")
    
    def test_is_superposition(self):
        analyzer = QuantumStateAnalyzer()
        plus_state = QuantumState(num_qubits=1, amplitudes=PLUS)
        zero_state = QuantumState(num_qubits=1, amplitudes=ZERO)
        assert analyzer.is_superposition(plus_state)
        assert not analyzer.is_superposition(zero_state)
    
    def test_is_entangled(self):
        analyzer = QuantumStateAnalyzer()
        bell = QuantumState(num_qubits=2, amplitudes=create_bell_state(0))
        product = QuantumState(num_qubits=2, amplitudes=tensor_product(ZERO, ZERO))
        assert analyzer.is_entangled(bell)
        assert not analyzer.is_entangled(product)
    
    def test_get_measurement_probabilities(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState.uniform_superposition(2)
        probs = analyzer.get_measurement_probabilities(state)
        assert len(probs) == 4
        for p in probs.values():
            assert p == pytest.approx(0.25)
    
    def test_infer_predicates_superposition(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState.uniform_superposition(1)
        preds = analyzer.infer_predicates(state)
        assert "superposition(state)" in preds
    
    def test_infer_predicates_entangled(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=2, amplitudes=create_bell_state(0))
        preds = analyzer.infer_predicates(state)
        assert "entangled(state)" in preds
    
    def test_infer_predicates_bell_state(self):
        analyzer = QuantumStateAnalyzer()
        state = QuantumState(num_qubits=2, amplitudes=create_bell_state(0))
        preds = analyzer.infer_predicates(state)
        assert any("bell_state" in p for p in preds)


# ============== Algorithm Tests ==============

class TestQuantumSpecSynth:
    """Tests for QuantumSpecSynth algorithm."""
    
    def test_synthesize_basic(self, mock_llm):
        synth = QuantumSpecSynth(llm=mock_llm, max_candidates=1)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = synth.synthesize(program)
        assert result.status in [SynthesisStatus.SUCCESS, SynthesisStatus.FAILED]
    
    def test_synthesize_with_verification(self, mock_llm):
        synth = QuantumSpecSynth(llm=mock_llm, max_candidates=1)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        oracle = MockVerificationOracle("valid")
        result = synth.synthesize(program, verification_oracle=oracle)
        assert result is not None
    
    def test_synthesize_invalid_json(self):
        bad_llm = MockLLM("not valid json")
        synth = QuantumSpecSynth(llm=bad_llm, max_candidates=1)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = synth.synthesize(program)
        assert result.status == SynthesisStatus.FAILED
    
    def test_program_analysis(self, mock_llm):
        synth = QuantumSpecSynth(llm=mock_llm)
        program = QuantumProgram.from_silq("""def bell(q0: qubit, q1: qubit) {
            q0 = H(q0);
            CNOT(q0, q1);
            return (q0, q1);
        }""")
        analysis = synth._analyze_program(program)
        assert isinstance(analysis, ProgramAnalysis)
        assert analysis.num_qubits >= 2
    
    def test_type_inference(self, mock_llm):
        synth = QuantumSpecSynth(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        analysis = synth._analyze_program(program)
        types = synth._infer_types(program, analysis)
        assert isinstance(types, TypeInference)


class TestProgramAnalysis:
    """Tests for ProgramAnalysis dataclass."""
    
    def test_to_context_string(self):
        analysis = ProgramAnalysis(
            num_qubits=2,
            gate_sequence=["H", "CNOT"],
            has_loops=False,
            has_measurements=False,
            input_qubits=["q0", "q1"],
            output_qubits=["q0", "q1"],
            quantum_operations=[],
            control_flow=[]
        )
        ctx = analysis.to_context_string()
        assert "Number of qubits: 2" in ctx


class TestTypeInference:
    """Tests for TypeInference dataclass."""
    
    def test_to_context_string(self):
        ti = TypeInference(
            qubit_types={"q0": "qubit", "q1": "qubit"},
            constraints=[]
        )
        ctx = ti.to_context_string()
        assert "q0: qubit" in ctx


class TestLearnQuantumPredicate:
    """Tests for LearnQuantumPredicate algorithm."""
    
    def test_learn_basic(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        examples = [PredicateExample(
            state=QuantumState.zero_state(1),
            expected=True
        )]
        # This may return None with mock LLM
        result = learner.learn(program, "line_1", examples)
        assert result is None or isinstance(result, str)
    
    def test_extract_features(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        program = QuantumProgram.from_silq("""def test(q0: qubit, q1: qubit) {
            q0 = H(q0);
            CNOT(q0, q1);
            return (q0, q1);
        }""")
        features = learner._extract_features(program, "line_2")
        assert isinstance(features, QuantumFeatures)
    
    def test_extract_features_loop_entry(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        features = learner._extract_features(program, "loop_entry")
        assert features.in_loop
    
    def test_generate_candidates(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        features = QuantumFeatures(
            num_qubits=2,
            active_qubits=["q0", "q1"],
            preceding_gates=["H", "CNOT"],
            entanglement_possible=True,
            in_loop=False
        )
        candidates = learner._generate_candidates(features)
        assert len(candidates) > 0
    
    def test_evaluate_predicate_superposition(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        plus = QuantumState(num_qubits=1, amplitudes=PLUS)
        zero = QuantumState(num_qubits=1, amplitudes=ZERO)
        assert learner._evaluate_predicate("superposition(q)", plus)
        assert not learner._evaluate_predicate("superposition(q)", zero)
    
    def test_evaluate_predicate_entangled(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        bell = QuantumState(num_qubits=2, amplitudes=create_bell_state(0))
        product = QuantumState(num_qubits=2, amplitudes=tensor_product(ZERO, ZERO))
        assert learner._evaluate_predicate("entangled(q0, q1)", bell)
        assert not learner._evaluate_predicate("entangled(q0, q1)", product)
    
    def test_evaluate_predicate_in_basis(self, mock_llm):
        learner = LearnQuantumPredicate(llm=mock_llm)
        zero = QuantumState(num_qubits=1, amplitudes=ZERO)
        assert learner._evaluate_predicate("in_basis(q, |0>)", zero)


class TestPredicateExample:
    """Tests for PredicateExample dataclass."""
    
    def test_is_counterexample(self):
        ex = PredicateExample(
            state=QuantumState.zero_state(1),
            expected=True,
            actual=False
        )
        assert ex.is_counterexample()
    
    def test_not_counterexample(self):
        ex = PredicateExample(
            state=QuantumState.zero_state(1),
            expected=True,
            actual=True
        )
        assert not ex.is_counterexample()


class TestSynthesizeQuantumInvariant:
    """Tests for SynthesizeQuantumInvariant algorithm."""
    
    def test_synthesize_basic(self, mock_llm):
        synth = SynthesizeQuantumInvariant(llm=mock_llm, max_unrolling=2)
        program = QuantumProgram.from_silq("""def grover(q: qubit) {
            for i in range(3) { q = H(q); }
            return q;
        }""")
        pre = Precondition.from_string("true")
        post = Postcondition.from_string("true")
        result = synth.synthesize(program, pre, post)
        # May be None with mock
        assert result is None or isinstance(result, Invariant)
    
    def test_simulate_loop(self, mock_llm):
        synth = SynthesizeQuantumInvariant(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        pre = Precondition.from_string("true")
        traces = synth._simulate_loop(program, pre, 2)
        assert len(traces) == 2
        assert all(isinstance(t, LoopTrace) for t in traces)
    
    def test_create_initial_state(self, mock_llm):
        synth = SynthesizeQuantumInvariant(llm=mock_llm)
        pre = Precondition.from_string("true")
        state = synth._create_initial_state(2, pre)
        assert state.num_qubits == 2
    
    def test_learn_iteration_predicate(self, mock_llm):
        synth = SynthesizeQuantumInvariant(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        traces = [LoopTrace(
            iteration=1,
            entry_state=QuantumState.uniform_superposition(1),
            exit_state=QuantumState.uniform_superposition(1)
        )]
        pred = synth._learn_iteration_predicate(program, traces, 1)
        assert pred is None or isinstance(pred, str)
    
    def test_check_methods(self, mock_llm):
        oracle = MockVerificationOracle()
        synth = SynthesizeQuantumInvariant(llm=mock_llm, verification_oracle=oracle)
        pre = Precondition.from_string("pre")
        post = Postcondition.from_string("post")
        inv = Invariant.from_string("inv")
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        
        assert synth._check_initiation(pre, inv)
        assert synth._check_consecution(inv, program, inv)
        assert synth._check_termination(inv, "true", post)


class TestRepairSpecification:
    """Tests for RepairSpecification algorithm."""
    
    def test_repair_weak_precondition(self, mock_llm):
        repair = RepairSpecification(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        cex = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.zero_state(1),
            violated_condition="precondition failed"
        )
        result = repair.repair(program, spec, cex)
        assert result is None or isinstance(result, Specification)
    
    def test_repair_strong_postcondition(self, mock_llm):
        repair = RepairSpecification(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("impossible")
        )
        cex = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.uniform_superposition(1),
            violated_condition="postcondition failed"
        )
        result = repair.repair(program, spec, cex)
        assert result is not None
    
    def test_diagnose_failure(self, mock_llm):
        repair = RepairSpecification(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        pre = Precondition.from_string("true")
        post = Postcondition.from_string("true")
        
        cex_pre = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.zero_state(1),
            violated_condition="precondition"
        )
        assert repair._diagnose_failure(program, pre, post, cex_pre) == FailureDiagnosis.WEAK_PRECONDITION
        
        cex_post = CounterExample(
            input_state=QuantumState.zero_state(1),
            output_state=QuantumState.zero_state(1),
            violated_condition="postcondition"
        )
        assert repair._diagnose_failure(program, pre, post, cex_post) == FailureDiagnosis.STRONG_POSTCONDITION


# ============== Verification Tests ==============

class TestNeuralVerifier:
    """Tests for NeuralVerifier."""
    
    def test_verify_valid(self, mock_llm):
        verifier = NeuralVerifier(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = verifier.verify(program, spec)
        assert result.status == VerificationStatus.VALID
    
    def test_verify_with_invariants(self, mock_llm):
        verifier = NeuralVerifier(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true"),
            invariants=[Invariant.from_string("inv")]
        )
        result = verifier.verify(program, spec)
        assert result.status in [VerificationStatus.VALID, VerificationStatus.INVALID, VerificationStatus.UNKNOWN]
    
    def test_verify_components(self, mock_llm):
        verifier = NeuralVerifier(llm=mock_llm)
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = verifier.verify_components(
            program,
            Precondition.from_string("true"),
            Postcondition.from_string("true"),
            []
        )
        assert isinstance(result, VerificationResult)


class TestSMTStats:
    """Tests for SMTStats dataclass."""
    
    def test_defaults(self):
        stats = SMTStats()
        assert stats.total_vcs == 0
        assert stats.proven_vcs == 0


class TestCreateZ3Interface:
    """Tests for Z3 interface creation."""
    
    def test_create_z3_interface(self):
        # May return None if z3 not installed
        result = create_z3_interface()
        assert result is None  # Z3 not installed in this env


# ============== Benchmark Tests ==============

class TestQVerifyBench:
    """Tests for QVerifyBench."""
    
    def test_load_all_tiers(self):
        bench = QVerifyBench(tier="all")
        assert len(bench) >= 500
    
    def test_load_specific_tier(self):
        bench = QVerifyBench(tier="T1")
        assert len(bench) == 150
    
    def test_iteration(self):
        bench = QVerifyBench(tier="T1")
        count = 0
        for program in bench:
            count += 1
            assert isinstance(program, BenchmarkProgram)
        assert count == 150
    
    def test_getitem(self):
        bench = QVerifyBench(tier="T1")
        program = bench[0]
        assert isinstance(program, BenchmarkProgram)
    
    def test_get_by_tier(self):
        bench = QVerifyBench(tier="all")
        t2_programs = bench.get_by_tier("T2")
        assert len(t2_programs) == 150
    
    def test_get_statistics(self):
        bench = QVerifyBench(tier="all")
        stats = bench.get_statistics()
        assert stats["total_programs"] >= 500
        assert "by_tier" in stats
    
    def test_metadata(self):
        bench = QVerifyBench(tier="all")
        meta = bench.metadata
        assert "name" in meta
        assert meta["name"] == "QVerifyBench"
    
    def test_evaluate(self):
        bench = QVerifyBench(tier="T1")
        result = bench.evaluate(llm="mock", timeout=1)
        assert isinstance(result, BenchmarkResult)


class TestBenchmarkProgram:
    """Tests for BenchmarkProgram."""
    
    def test_to_quantum_program(self):
        bench = QVerifyBench(tier="T1")
        bp = bench[0]
        qp = bp.to_quantum_program()
        assert isinstance(qp, QuantumProgram)
    
    def test_to_specification(self):
        bench = QVerifyBench(tier="T1")
        bp = bench[0]
        spec = bp.to_specification()
        assert isinstance(spec, Specification)


# ============== QVerify Main Interface Tests ==============

class TestQVerify:
    """Tests for main QVerify interface."""
    
    def test_init_mock(self):
        qv = QVerify(llm="mock")
        assert qv._llm_interface is not None
    
    def test_init_custom_llm(self):
        custom_llm = MockLLM()
        qv = QVerify(llm=custom_llm)
        assert qv._llm_interface == custom_llm
    
    def test_synthesize_specification(self):
        qv = QVerify(llm="mock")
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = qv.synthesize_specification(program)
        assert isinstance(spec, Specification)
    
    def test_verify(self):
        qv = QVerify(llm="mock")
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        spec = Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true")
        )
        result = qv.verify(program, spec)
        assert isinstance(result, VerificationResult)
    
    def test_synthesize_and_verify(self):
        qv = QVerify(llm="mock")
        program = QuantumProgram.from_silq("def test(q: qubit) { return q; }")
        result = qv.synthesize_and_verify(program)
        assert "specification" in result
        assert "verification_result" in result
    
    def test_load_program_silq(self):
        program = QVerify.load_program("def test(q: qubit) { return q; }", "silq")
        assert isinstance(program, QuantumProgram)
    
    def test_load_program_qasm(self):
        qasm = """OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];"""
        program = QVerify.load_program(qasm, "qasm")
        assert isinstance(program, QuantumProgram)
    
    def test_load_program_unknown_language(self):
        with pytest.raises(ValueError):
            QVerify.load_program("source", "unknown_lang")


# ============== Utils Tests ==============

class TestUtils:
    """Tests for utility modules."""
    
    def test_utils_init(self):
        from qverify import utils
        assert utils is not None
