"""LearnQuantumPredicate: Neural Predicate Synthesis for Quantum States."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from qverify.core.quantum_program import QuantumProgram
from qverify.core.types import QuantumState

logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate completions."""
        ...


@dataclass
class QuantumFeatures:
    """Features extracted from a quantum program at a specific point."""

    num_qubits: int
    active_qubits: List[str]
    preceding_gates: List[str]
    entanglement_possible: bool
    in_loop: bool
    loop_iteration: Optional[int] = None


@dataclass
class PredicateExample:
    """An example for predicate learning."""

    state: QuantumState
    expected: bool
    actual: Optional[bool] = None

    def is_counterexample(self) -> bool:
        """Check if this is a counterexample."""
        return self.actual is not None and self.actual != self.expected


PREDICATE_SYNTHESIS_PROMPT = """You are an expert in quantum program analysis.

## Program Point
Location: {location}
Preceding gates: {preceding_gates}

## Features
- Number of qubits: {num_qubits}
- Active qubits: {active_qubits}
- Entanglement possible: {entanglement_possible}

## Task
Synthesize a predicate that holds at this program point.

Output a single predicate formula using: in_basis, superposition, entangled, prob.

Predicate: """


@dataclass
class LearnQuantumPredicate:
    """Learn quantum state predicates from execution traces."""

    llm: LLMInterface
    max_iterations: int = 5
    tolerance: float = 1e-10

    def learn(
        self,
        program: QuantumProgram,
        location: str,
        examples: List[PredicateExample],
    ) -> Optional[str]:
        """Learn a predicate that holds at the given program location."""
        features = self._extract_features(program, location)
        candidates = self._generate_candidates(features)

        for iteration in range(self.max_iterations):
            prompt = self._build_prompt(program, location, examples, features, candidates)
            responses = self.llm.generate(prompt, n=1)
            if not responses:
                continue

            predicate = self._parse_predicate(responses[0])
            if predicate is None:
                continue

            all_correct = True
            for example in examples:
                actual = self._evaluate_predicate(predicate, example.state)
                example.actual = actual
                if actual != example.expected:
                    all_correct = False
                    break

            if all_correct:
                return predicate

        return None

    def _extract_features(self, program: QuantumProgram, location: str) -> QuantumFeatures:
        """Extract features at a program point."""
        preceding_gates = []
        in_loop = False

        if location.startswith("line_"):
            try:
                line_num = int(location.split("_")[1])
                for gate in program.gates:
                    if gate.line_number < line_num:
                        preceding_gates.append(gate.name)
            except (ValueError, IndexError):
                pass
        elif location == "loop_entry":
            in_loop = True
            for gate in program.gates:
                preceding_gates.append(gate.name)

        entangling_gates = {"CNOT", "CX", "CZ", "SWAP"}
        entanglement_possible = any(g in entangling_gates for g in preceding_gates)

        return QuantumFeatures(
            num_qubits=program.num_qubits,
            active_qubits=program.qubits,
            preceding_gates=preceding_gates,
            entanglement_possible=entanglement_possible,
            in_loop=in_loop,
        )

    def _generate_candidates(self, features: QuantumFeatures) -> List[str]:
        """Generate candidate predicate templates."""
        candidates = []
        for qubit in features.active_qubits:
            candidates.append(f"superposition({qubit})")
            candidates.append(f"in_basis({qubit}, |0>)")
        if features.entanglement_possible and len(features.active_qubits) >= 2:
            for i, q1 in enumerate(features.active_qubits):
                for q2 in features.active_qubits[i + 1:]:
                    candidates.append(f"entangled({q1}, {q2})")
        return candidates

    def _build_prompt(self, program: QuantumProgram, location: str, examples: List[PredicateExample], features: QuantumFeatures, candidates: List[str]) -> str:
        """Build LLM prompt for predicate synthesis."""
        return PREDICATE_SYNTHESIS_PROMPT.format(
            location=location,
            preceding_gates=" -> ".join(features.preceding_gates[-10:]),
            num_qubits=features.num_qubits,
            active_qubits=", ".join(features.active_qubits),
            entanglement_possible=features.entanglement_possible,
        )

    def _parse_predicate(self, response: str) -> Optional[str]:
        """Parse predicate from LLM response."""
        response = response.strip()
        if "Predicate:" in response:
            response = response.split("Predicate:")[-1].strip()
        response = response.strip("`").strip()
        valid_predicates = ["in_basis", "superposition", "entangled", "prob", "amplitude"]
        if any(pred in response for pred in valid_predicates):
            return response
        return None

    def _evaluate_predicate(self, predicate: str, state: QuantumState) -> bool:
        """Evaluate a predicate on a quantum state."""
        predicate = predicate.lower()
        if "superposition" in predicate:
            nonzero_count = sum(1 for amp in state.amplitudes if abs(amp) > self.tolerance)
            return nonzero_count > 1
        if "entangled" in predicate:
            if state.num_qubits < 2:
                return False
            matrix = state.amplitudes.reshape(2, -1)
            _, s, _ = np.linalg.svd(matrix)
            return sum(sv > self.tolerance for sv in s) > 1
        if "in_basis" in predicate:
            if "|0>" in predicate or "zero" in predicate:
                return abs(state.amplitudes[0]) > 1 - self.tolerance
        return True
