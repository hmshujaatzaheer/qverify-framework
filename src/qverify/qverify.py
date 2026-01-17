"""Main QVERIFY interface."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Postcondition, Precondition, Specification
from qverify.core.types import VerificationResult
from qverify.algorithms.spec_synth import QuantumSpecSynth
from qverify.verification.neural_verifier import NeuralVerifier

logger = logging.getLogger(__name__)


class MockLLM:
    """Mock LLM for testing without API keys."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate mock specifications."""
        if "bell" in prompt.lower() or "entangle" in prompt.lower():
            spec = {"precondition": "in_basis(q0, |0>) and in_basis(q1, |0>)", "postcondition": "entangled(q0, q1)", "invariants": []}
        elif "grover" in prompt.lower():
            spec = {"precondition": "superposition(qubits)", "postcondition": "amplified(qubits)", "invariants": ["prob(marked) >= 1/N"]}
        elif "hadamard" in prompt.lower():
            spec = {"precondition": "in_basis(q, |0>)", "postcondition": "superposition(q)", "invariants": []}
        else:
            spec = {"precondition": "true", "postcondition": "true", "invariants": []}
        return [json.dumps(spec)] * n


@dataclass
class QVerify:
    """Main QVERIFY interface for quantum program verification."""

    llm: Union[str, Any] = "mock"
    backend: str = "z3"
    timeout: float = 30.0
    enable_refinement: bool = True
    verbose: bool = False
    _llm_interface: Any = field(default=None, init=False)
    _verifier: Optional[NeuralVerifier] = field(default=None, init=False)
    _synthesizer: Optional[QuantumSpecSynth] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize components."""
        self._setup_llm()
        self._setup_verifier()
        self._setup_synthesizer()

    def _setup_llm(self):
        """Initialize LLM interface."""
        if isinstance(self.llm, str):
            if self.llm == "mock":
                self._llm_interface = MockLLM()
            else:
                self._llm_interface = MockLLM()
        else:
            self._llm_interface = self.llm

    def _setup_verifier(self):
        """Initialize verification engine."""
        self._verifier = NeuralVerifier(llm=self._llm_interface, timeout=self.timeout)

    def _setup_synthesizer(self):
        """Initialize specification synthesizer."""
        self._synthesizer = QuantumSpecSynth(llm=self._llm_interface, max_candidates=5)

    def synthesize_specification(self, program: QuantumProgram, hints: Optional[Dict[str, str]] = None) -> Specification:
        """Synthesize a specification for a quantum program."""
        verification_oracle = self._verifier if self.enable_refinement else None
        result = self._synthesizer.synthesize(program, verification_oracle=verification_oracle, enable_refinement=self.enable_refinement)

        if result.is_success() and result.specification:
            return result.specification

        return Specification(
            precondition=Precondition.from_string("true"),
            postcondition=Postcondition.from_string("true"),
            name="minimal_spec",
        )

    def verify(self, program: QuantumProgram, specification: Specification) -> VerificationResult:
        """Verify a quantum program against a specification."""
        return self._verifier.verify(program, specification)

    def synthesize_and_verify(self, program: QuantumProgram) -> Dict[str, Any]:
        """Synthesize specification and verify in one step."""
        spec = self.synthesize_specification(program)
        result = self.verify(program, spec)
        return {"specification": spec, "verification_result": result, "success": result.is_valid()}

    @staticmethod
    def load_program(source: str, language: str = "silq") -> QuantumProgram:
        """Load a quantum program from source code."""
        if language.lower() == "silq":
            return QuantumProgram.from_silq(source)
        elif language.lower() in ("qasm", "openqasm"):
            return QuantumProgram.from_qasm(source)
        else:
            raise ValueError(f"Unknown language: {language}")
