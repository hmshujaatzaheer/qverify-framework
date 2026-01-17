"""RepairSpecification: Counterexample-Guided Specification Refinement."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Protocol

import numpy as np

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Postcondition, Precondition, Specification
from qverify.core.types import CounterExample

logger = logging.getLogger(__name__)


class FailureDiagnosis(Enum):
    """Types of specification failures."""

    WEAK_PRECONDITION = auto()
    STRONG_POSTCONDITION = auto()
    LOGICAL_ERROR = auto()
    UNKNOWN = auto()


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate completions."""
        ...


REPAIR_PROMPT = """You are an expert in quantum program specification.

## Failed Specification
Precondition: {precondition}
Postcondition: {postcondition}

## Counterexample
Violated condition: {cex_violated}

## Diagnosis
{diagnosis_explanation}

## Task
Repair the specification.

```json
{{
    "precondition": "repaired precondition",
    "postcondition": "repaired postcondition",
    "reasoning": "why this fixes the issue"
}}
```"""


@dataclass
class RepairSpecification:
    """Repair failed specifications using counterexample guidance."""

    llm: LLMInterface
    max_repair_attempts: int = 3

    def repair(
        self,
        program: QuantumProgram,
        failed_spec: Specification,
        counterexample: CounterExample,
    ) -> Optional[Specification]:
        """Repair a failed specification."""
        diagnosis = self._diagnose_failure(
            program, failed_spec.precondition, failed_spec.postcondition, counterexample
        )

        if diagnosis == FailureDiagnosis.WEAK_PRECONDITION:
            new_pre = self._strengthen_precondition(failed_spec.precondition, counterexample)
            return Specification(
                precondition=new_pre,
                postcondition=failed_spec.postcondition,
                invariants=failed_spec.invariants,
                name=f"{failed_spec.name}_repaired",
            )

        elif diagnosis == FailureDiagnosis.STRONG_POSTCONDITION:
            new_post = self._weaken_postcondition(failed_spec.postcondition, counterexample)
            return Specification(
                precondition=failed_spec.precondition,
                postcondition=new_post,
                invariants=failed_spec.invariants,
                name=f"{failed_spec.name}_repaired",
            )

        else:
            return self._llm_repair(program, failed_spec, counterexample, diagnosis)

    def _diagnose_failure(self, program, pre: Precondition, post: Postcondition, cex: CounterExample) -> FailureDiagnosis:
        """Diagnose why the specification failed."""
        violated = cex.violated_condition.lower()

        if "precondition" in violated or "pre" in violated:
            return FailureDiagnosis.WEAK_PRECONDITION

        if "postcondition" in violated or "post" in violated:
            return FailureDiagnosis.STRONG_POSTCONDITION

        input_entropy = self._state_entropy(cex.input_state)
        if input_entropy < 0.1:
            return FailureDiagnosis.WEAK_PRECONDITION

        output_entropy = self._state_entropy(cex.output_state)
        if output_entropy > 0.5:
            return FailureDiagnosis.STRONG_POSTCONDITION

        return FailureDiagnosis.UNKNOWN

    def _state_entropy(self, state) -> float:
        """Calculate entropy of a quantum state."""
        probs = np.abs(state.amplitudes) ** 2
        probs = probs[probs > 1e-10]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _strengthen_precondition(self, pre: Precondition, cex: CounterExample) -> Precondition:
        """Strengthen precondition to exclude the counterexample input."""
        exclusion = self._generate_exclusion(cex.input_state)
        new_formula = f"({pre.formula}) and not({exclusion})"
        return Precondition.from_string(new_formula)

    def _weaken_postcondition(self, post: Postcondition, cex: CounterExample) -> Postcondition:
        """Weaken postcondition to allow the counterexample output."""
        allowed = self._generate_allowed(cex.output_state)
        new_formula = f"({post.formula}) or ({allowed})"
        return Postcondition.from_string(new_formula)

    def _generate_exclusion(self, state) -> str:
        """Generate a formula that excludes a specific input state."""
        max_idx = int(np.argmax(np.abs(state.amplitudes) ** 2))
        basis = format(max_idx, f'0{state.num_qubits}b')
        conditions = []
        for i, bit in enumerate(basis):
            qubit = f"q{i}"
            if bit == '0':
                conditions.append(f"in_basis({qubit}, |0>)")
            else:
                conditions.append(f"in_basis({qubit}, |1>)")
        return " and ".join(conditions)

    def _generate_allowed(self, state) -> str:
        """Generate a formula that allows a specific output state."""
        nonzero = sum(1 for amp in state.amplitudes if abs(amp) > 1e-10)
        if nonzero > 1:
            return "superposition(qubits)"
        else:
            max_idx = int(np.argmax(np.abs(state.amplitudes) ** 2))
            basis = format(max_idx, f'0{state.num_qubits}b')
            return f"in_basis(qubits, |{basis}>)"

    def _llm_repair(self, program: QuantumProgram, failed_spec: Specification, cex: CounterExample, diagnosis: FailureDiagnosis) -> Optional[Specification]:
        """Use LLM for complex specification repairs."""
        diagnosis_explanations = {
            FailureDiagnosis.WEAK_PRECONDITION: "The precondition is too weak.",
            FailureDiagnosis.STRONG_POSTCONDITION: "The postcondition is too strong.",
            FailureDiagnosis.LOGICAL_ERROR: "There is a logical error.",
            FailureDiagnosis.UNKNOWN: "The cause could not be determined.",
        }

        prompt = REPAIR_PROMPT.format(
            precondition=failed_spec.precondition.formula,
            postcondition=failed_spec.postcondition.formula,
            cex_violated=cex.violated_condition,
            diagnosis_explanation=diagnosis_explanations.get(diagnosis, "Unknown"),
        )

        responses = self.llm.generate(prompt, n=1)
        if not responses:
            return None

        try:
            response = responses[0]
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                return Specification.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            pass

        return None
