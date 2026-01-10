"""
QuantumSpecSynth: LLM-Guided Specification Synthesis for Quantum Programs.

This module implements Algorithm 1 from the QVERIFY paper - the core specification
synthesis algorithm that uses LLMs to generate formal specifications for quantum
programs with counterexample-guided refinement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import (
    Specification,
    Precondition,
    Postcondition,
    Invariant,
)
from qverify.core.types import (
    SynthesisResult,
    SynthesisStatus,
    VerificationResult,
    VerificationStatus,
    CounterExample,
)


logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate n completions for the given prompt."""
        ...


class VerificationOracle(Protocol):
    """Protocol for verification oracle."""
    
    def verify(
        self,
        program: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        invariants: list[Invariant],
    ) -> VerificationResult:
        """Verify program against specification."""
        ...


@dataclass
class ProgramAnalysis:
    """Results of analyzing a quantum program."""
    
    num_qubits: int
    gate_sequence: list[str]
    has_loops: bool
    has_measurements: bool
    input_qubits: list[str]
    output_qubits: list[str]
    quantum_operations: list[dict[str, Any]]
    control_flow: list[dict[str, Any]]
    
    def to_context_string(self) -> str:
        """Convert analysis to context string for LLM prompt."""
        lines = [
            f"Number of qubits: {self.num_qubits}",
            f"Input qubits: {', '.join(self.input_qubits)}",
            f"Output qubits: {', '.join(self.output_qubits)}",
            f"Gate sequence: {' -> '.join(self.gate_sequence[:20])}",  # Limit length
            f"Has loops: {self.has_loops}",
            f"Has measurements: {self.has_measurements}",
        ]
        return "\n".join(lines)


@dataclass
class TypeInference:
    """Quantum type information for a program."""
    
    qubit_types: dict[str, str]  # qubit_name -> type (qubit, qubit[], etc.)
    constraints: list[str]  # Type constraints discovered
    
    def to_context_string(self) -> str:
        """Convert type info to context string."""
        type_lines = [f"  {name}: {typ}" for name, typ in self.qubit_types.items()]
        return "Qubit types:\n" + "\n".join(type_lines)


# Prompt templates
SPEC_SYNTHESIS_PROMPT = """You are an expert in quantum program verification. Your task is to synthesize a formal specification for the following quantum program.

## Quantum Program
```
{program_source}
```

## Program Analysis
{program_analysis}

## Type Information
{type_info}

## Your Task
Generate a formal specification consisting of:
1. **Precondition**: What must be true about the input quantum state before execution
2. **Postcondition**: What is guaranteed about the output quantum state after execution
3. **Invariants** (if loops present): What holds at each loop iteration

## Available Predicates
- `in_basis(qubit, state)`: Qubit is in specified basis state (|0⟩, |1⟩, |+⟩, |−⟩)
- `superposition(qubit)`: Qubit is in a non-trivial superposition
- `entangled(q1, q2)`: Qubits are entangled
- `prob(qubit, outcome) op value`: Probability constraint (op: =, >=, <=, >, <)
- `amplitude(qubit, basis) op value`: Amplitude constraint

## Output Format
Provide your answer as JSON:
```json
{{
    "precondition": "your precondition formula",
    "postcondition": "your postcondition formula",
    "invariants": ["invariant1", "invariant2"],
    "reasoning": "brief explanation of your specification"
}}
```

Generate the specification:"""


REFINEMENT_PROMPT = """The previous specification was invalid. Here is the counterexample:

## Previous Specification
Precondition: {prev_pre}
Postcondition: {prev_post}

## Counterexample
{counterexample}

## Error Analysis
The specification failed because: {error_reason}

## Your Task
Refine the specification to handle this counterexample. Either:
1. Strengthen the precondition to exclude this input
2. Weaken the postcondition to allow this output
3. Fix the logical error in the specification

Provide the refined specification as JSON:
```json
{{
    "precondition": "refined precondition",
    "postcondition": "refined postcondition",
    "invariants": ["refined invariants"],
    "reasoning": "why this refinement fixes the issue"
}}
```"""


@dataclass
class QuantumSpecSynth:
    """
    LLM-guided specification synthesis for quantum programs.
    
    This class implements the QuantumSpecSynth algorithm that uses large language
    models to synthesize formal specifications for quantum programs. It supports
    counterexample-guided refinement to iteratively improve specifications.
    
    Example:
        >>> from qverify.algorithms import QuantumSpecSynth
        >>> synth = QuantumSpecSynth(llm=my_llm, max_candidates=5)
        >>> spec = synth.synthesize(program, verifier)
        >>> print(spec.precondition)
    """
    
    llm: LLMInterface
    max_candidates: int = 5
    max_refinement_iterations: int = 3
    timeout_seconds: float = 60.0
    enable_type_guidance: bool = True
    enable_quantum_aware_prompting: bool = True
    
    # Statistics
    _llm_calls: int = field(default=0, init=False, repr=False)
    _candidates_tried: int = field(default=0, init=False, repr=False)
    
    def synthesize(
        self,
        program: QuantumProgram,
        verification_oracle: Optional[VerificationOracle] = None,
        enable_refinement: bool = True,
    ) -> SynthesisResult:
        """
        Synthesize a specification for the given quantum program.
        
        Args:
            program: The quantum program to synthesize a specification for
            verification_oracle: Optional oracle for verifying candidate specs
            enable_refinement: Whether to use counterexample-guided refinement
            
        Returns:
            SynthesisResult containing the synthesized specification or failure info
        """
        start_time = time.time()
        self._llm_calls = 0
        self._candidates_tried = 0
        refinement_iterations = 0
        
        try:
            # Step 1: Analyze program
            analysis = self._analyze_program(program)
            
            # Step 2: Infer types
            type_info = self._infer_types(program, analysis)
            
            # Step 3: Build context for LLM
            context = self._extract_context(program, analysis, type_info)
            
            # Step 4: Build initial prompt
            prompt = self._build_prompt(context)
            
            # Step 5: Generate candidates
            candidates = self._generate_candidates(prompt)
            self._llm_calls += 1
            
            # Step 6: Try each candidate
            for candidate_str in candidates:
                self._candidates_tried += 1
                
                # Parse candidate
                spec = self._parse_specification(candidate_str)
                if spec is None:
                    continue
                
                # If no oracle, return first valid parse
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
                
                # Verify specification
                result = verification_oracle.verify(
                    program,
                    spec.precondition,
                    spec.postcondition,
                    spec.invariants,
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
                
                # Try refinement if enabled
                if enable_refinement and result.counterexample is not None:
                    refined_spec = self._refine_with_counterexample(
                        program, spec, result.counterexample, context
                    )
                    refinement_iterations += 1
                    
                    if refined_spec is not None:
                        # Verify refined spec
                        result = verification_oracle.verify(
                            program,
                            refined_spec.precondition,
                            refined_spec.postcondition,
                            refined_spec.invariants,
                        )
                        
                        if result.status == VerificationStatus.VALID:
                            return SynthesisResult(
                                status=SynthesisStatus.SUCCESS,
                                specification=refined_spec,
                                candidates_tried=self._candidates_tried,
                                refinement_iterations=refinement_iterations,
                                time_seconds=time.time() - start_time,
                                llm_calls=self._llm_calls,
                                message="Specification refined and verified"
                            )
            
            # All candidates failed
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
        output_qubits = input_qubits.copy()  # Simplified
        
        quantum_ops = []
        for gate in program.gates:
            quantum_ops.append({
                "gate": gate.name,
                "qubits": [str(q) for q in gate.qubits],
                "line": gate.line_number,
            })
        
        return ProgramAnalysis(
            num_qubits=program.num_qubits,
            gate_sequence=gate_sequence,
            has_loops=program.has_loops,
            has_measurements=program.has_measurements,
            input_qubits=input_qubits,
            output_qubits=output_qubits,
            quantum_operations=quantum_ops,
            control_flow=[],
        )
    
    def _infer_types(
        self, 
        program: QuantumProgram, 
        analysis: ProgramAnalysis
    ) -> TypeInference:
        """Infer quantum types for program variables."""
        qubit_types = {}
        constraints = []
        
        for qubit in program.qubits:
            qubit_types[str(qubit)] = "qubit"
        
        # Add constraints based on operations
        for op in analysis.quantum_operations:
            if op["gate"] in {"CNOT", "CZ", "SWAP"}:
                if len(op["qubits"]) >= 2:
                    constraints.append(
                        f"entanglement_possible({op['qubits'][0]}, {op['qubits'][1]})"
                    )
        
        return TypeInference(qubit_types=qubit_types, constraints=constraints)
    
    def _extract_context(
        self,
        program: QuantumProgram,
        analysis: ProgramAnalysis,
        type_info: TypeInference,
    ) -> dict[str, Any]:
        """Extract context for LLM prompting."""
        return {
            "program_source": program.source,
            "program_analysis": analysis.to_context_string(),
            "type_info": type_info.to_context_string(),
            "num_qubits": analysis.num_qubits,
            "has_loops": analysis.has_loops,
            "has_measurements": analysis.has_measurements,
        }
    
    def _build_prompt(self, context: dict[str, Any]) -> str:
        """Build prompt for LLM."""
        return SPEC_SYNTHESIS_PROMPT.format(**context)
    
    def _generate_candidates(self, prompt: str) -> list[str]:
        """Generate specification candidates using LLM."""
        return self.llm.generate(prompt, n=self.max_candidates)
    
    def _parse_specification(self, candidate_str: str) -> Optional[Specification]:
        """Parse a specification from LLM output."""
        try:
            # Extract JSON from response
            json_start = candidate_str.find('{')
            json_end = candidate_str.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            json_str = candidate_str[json_start:json_end]
            data = json.loads(json_str)
            
            # Create specification
            pre = Precondition.from_string(data.get("precondition", "true"))
            post = Postcondition.from_string(data.get("postcondition", "true"))
            
            invariants = []
            for inv_str in data.get("invariants", []):
                if inv_str and inv_str.strip():
                    invariants.append(Invariant.from_string(inv_str))
            
            return Specification(
                precondition=pre,
                postcondition=post,
                invariants=invariants,
                name="synthesized",
                description=data.get("reasoning", ""),
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug(f"Failed to parse specification: {e}")
            return None
    
    def _refine_with_counterexample(
        self,
        program: QuantumProgram,
        spec: Specification,
        counterexample: CounterExample,
        context: dict[str, Any],
    ) -> Optional[Specification]:
        """Refine specification using counterexample."""
        if self._llm_calls >= self.max_refinement_iterations + 1:
            return None
        
        prompt = REFINEMENT_PROMPT.format(
            prev_pre=spec.precondition.to_human_readable(),
            prev_post=spec.postcondition.to_human_readable(),
            counterexample=str(counterexample),
            error_reason=counterexample.violated_condition,
        )
        
        candidates = self.llm.generate(prompt, n=1)
        self._llm_calls += 1
        
        if candidates:
            return self._parse_specification(candidates[0])
        
        return None
    
    def _check_well_formed(
        self,
        pre: Precondition,
        post: Postcondition,
        invariants: list[Invariant],
        type_info: TypeInference,
    ) -> bool:
        """Check if specification is well-formed given type information."""
        # Basic well-formedness checks
        try:
            # Check precondition can be converted to SMT
            pre.to_smt()
            post.to_smt()
            for inv in invariants:
                inv.to_smt()
            return True
        except Exception:
            return False
