"""QuantumSpecSynth: LLM-Guided Specification Synthesis for Quantum Programs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Invariant, Postcondition, Precondition, Specification
from qverify.core.types import CounterExample, SynthesisResult, SynthesisStatus, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate n completions for the given prompt."""
        ...


class VerificationOracle(Protocol):
    """Protocol for verification oracle."""

    def verify(
        self,
        program: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        invariants: List[Invariant],
    ) -> VerificationResult:
        """Verify program against specification."""
        ...


@dataclass
class ProgramAnalysis:
    """Results of analyzing a quantum program."""

    num_qubits: int
    gate_sequence: List[str]
    has_loops: bool
    has_measurements: bool
    input_qubits: List[str]
    output_qubits: List[str]
    quantum_operations: List[Dict[str, Any]]
    control_flow: List[Dict[str, Any]]
    entanglement_pattern: str = "unknown"

    def to_context_string(self) -> str:
        """Convert analysis to context string for LLM prompt."""
        lines = [
            f"Number of qubits: {self.num_qubits}",
            f"Input qubits: {', '.join(self.input_qubits)}",
            f"Output qubits: {', '.join(self.output_qubits)}",
            f"Gate sequence: {' -> '.join(self.gate_sequence[:20])}",
            f"Has loops: {self.has_loops}",
            f"Has measurements: {self.has_measurements}",
            f"Entanglement pattern: {self.entanglement_pattern}",
        ]
        return "\n".join(lines)


@dataclass
class TypeInference:
    """Quantum type information for a program."""

    qubit_types: Dict[str, str]
    constraints: List[str]

    def to_context_string(self) -> str:
        """Convert type info to context string."""
        type_lines = [f"  {name}: {typ}" for name, typ in self.qubit_types.items()]
        return "Qubit types:\n" + "\n".join(type_lines)


SPEC_SYNTHESIS_PROMPT = """You are an expert in quantum program verification.

## Quantum Program
```
{program_source}
```

## Program Analysis
{program_analysis}

## Type Information
{type_info}

## Task
Generate a formal specification with precondition, postcondition, and invariants (if loops present).

## Output Format
```json
{{
    "precondition": "your precondition formula",
    "postcondition": "your postcondition formula",
    "invariants": ["invariant1"],
    "reasoning": "brief explanation"
}}
```
"""


@dataclass
class QuantumSpecSynth:
    """LLM-guided specification synthesis for quantum programs."""

    llm: LLMInterface
    max_candidates: int = 5
    max_refinement_iterations: int = 3
    timeout_seconds: float = 60.0
    enable_type_guidance: bool = True
    enable_quantum_aware_prompting: bool = True
    _llm_calls: int = field(default=0, init=False, repr=False)
    _candidates_tried: int = field(default=0, init=False, repr=False)

    def synthesize(
        self,
        program: QuantumProgram,
        verification_oracle: Optional[VerificationOracle] = None,
        enable_refinement: bool = True,
    ) -> SynthesisResult:
        """Synthesize a specification for the given quantum program."""
        start_time = time.time()
        self._llm_calls = 0
        self._candidates_tried = 0
        refinement_iterations = 0

        try:
            analysis = self._analyze_program(program)
            type_info = self._infer_types(program, analysis)
            context = self._extract_context(program, analysis, type_info)
            prompt = self._build_prompt(context)
            candidates = self._generate_candidates(prompt)
            self._llm_calls += 1

            for candidate in candidates:
                self._candidates_tried += 1
                spec = self._parse_specification(candidate)
                if spec is None:
                    continue

                if not self._check_well_formed(spec.precondition, spec.postcondition, spec.invariants, type_info):
                    continue

                if verification_oracle is None:
                    return SynthesisResult(
                        status=SynthesisStatus.SUCCESS,
                        specification=spec,
                        candidates_tried=self._candidates_tried,
                        refinement_iterations=refinement_iterations,
                        time_seconds=time.time() - start_time,
                        llm_calls=self._llm_calls,
                        message="Specification synthesized (not verified)"
                    )

                result = verification_oracle.verify(
                    program, spec.precondition, spec.postcondition, spec.invariants
                )

                if result.status == VerificationStatus.VALID:
                    return SynthesisResult(
                        status=SynthesisStatus.SUCCESS,
                        specification=spec,
                        candidates_tried=self._candidates_tried,
                        refinement_iterations=refinement_iterations,
                        time_seconds=time.time() - start_time,
                        llm_calls=self._llm_calls,
                        message="Specification synthesized and verified"
                    )

            return SynthesisResult(
                status=SynthesisStatus.FAILED,
                specification=None,
                candidates_tried=self._candidates_tried,
                refinement_iterations=refinement_iterations,
                time_seconds=time.time() - start_time,
                llm_calls=self._llm_calls,
                message="All candidate specifications failed verification"
            )

        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return SynthesisResult(
                status=SynthesisStatus.FAILED,
                specification=None,
                candidates_tried=self._candidates_tried,
                refinement_iterations=refinement_iterations,
                time_seconds=time.time() - start_time,
                llm_calls=self._llm_calls,
                message=f"Synthesis error: {str(e)}"
            )

    def _analyze_program(self, program: QuantumProgram) -> ProgramAnalysis:
        """Analyze quantum program structure."""
        gate_sequence = [str(g) for g in program.gates]
        input_qubits = [str(q) for q in program.qubits]
        output_qubits = input_qubits.copy()

        quantum_ops = []
        for gate in program.gates:
            quantum_ops.append({
                "gate": gate.name,
                "qubits": [str(q) for q in gate.qubits],
                "params": gate.params,
                "line": gate.line_number,
            })

        entangling_gates = [g for g in program.gates if len(g.qubits) >= 2]
        if not entangling_gates:
            entanglement = "none"
        elif len(entangling_gates) == 1:
            entanglement = "simple_pair"
        else:
            entanglement = "multi_qubit"

        return ProgramAnalysis(
            num_qubits=program.num_qubits,
            gate_sequence=gate_sequence,
            has_loops=program.has_loops,
            has_measurements=program.has_measurements,
            input_qubits=input_qubits,
            output_qubits=output_qubits,
            quantum_operations=quantum_ops,
            control_flow=[],
            entanglement_pattern=entanglement,
        )

    def _infer_types(self, program: QuantumProgram, analysis: ProgramAnalysis) -> TypeInference:
        """Infer quantum types for program variables."""
        qubit_types = {}
        constraints = []
        for qubit in program.qubits:
            qubit_types[str(qubit)] = "qubit"
        return TypeInference(qubit_types=qubit_types, constraints=constraints)

    def _extract_context(self, program: QuantumProgram, analysis: ProgramAnalysis, type_info: TypeInference) -> Dict[str, Any]:
        """Extract context for LLM prompting."""
        return {
            "program_source": program.source,
            "program_analysis": analysis.to_context_string(),
            "type_info": type_info.to_context_string(),
        }

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for LLM."""
        return SPEC_SYNTHESIS_PROMPT.format(**context)

    def _generate_candidates(self, prompt: str) -> List[str]:
        """Generate specification candidates using LLM."""
        return self.llm.generate(prompt, n=self.max_candidates)

    def _parse_specification(self, candidate_str: str) -> Optional[Specification]:
        """Parse a specification from LLM output."""
        try:
            json_start = candidate_str.find('{')
            json_end = candidate_str.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None
            json_str = candidate_str[json_start:json_end]
            data = json.loads(json_str)
            return Specification.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse specification: {e}")
            return None

    def _check_well_formed(self, pre: Precondition, post: Postcondition, invariants: List[Invariant], type_info: TypeInference) -> bool:
        """Check if specification is well-formed given type information."""
        try:
            pre.to_smt()
            post.to_smt()
            for inv in invariants:
                inv.to_smt()
            return True
        except Exception:
            return False
