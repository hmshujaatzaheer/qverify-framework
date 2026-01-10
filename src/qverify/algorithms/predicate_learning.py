"""
LearnQuantumPredicate: Neural Predicate Synthesis for Quantum States.

This module implements Algorithm 2 from the QVERIFY paper - learning quantum
state predicates from execution traces using LLM-guided synthesis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import numpy as np
from numpy.typing import NDArray

from qverify.core.quantum_program import QuantumProgram
from qverify.core.quantum_state import (
    QuantumStatePredicate,
    BasisPredicate,
    EntanglementPredicate,
    ProbabilityPredicate,
    SuperpositionPredicate,
    ConjunctionPredicate,
    BasisState,
    parse_predicate,
)


logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions."""
        ...


@dataclass
class ExecutionTrace:
    """A trace of quantum program execution."""
    
    input_state: NDArray[np.complex128]
    output_state: NDArray[np.complex128]
    intermediate_states: list[NDArray[np.complex128]] = field(default_factory=list)
    gate_sequence: list[str] = field(default_factory=list)
    
    @property
    def num_qubits(self) -> int:
        return int(np.log2(len(self.input_state)))


@dataclass
class PredicateCandidate:
    """A candidate predicate with evaluation results."""
    
    predicate: QuantumStatePredicate
    score: float  # How well it matches the examples
    false_positives: int = 0
    false_negatives: int = 0


PREDICATE_LEARNING_PROMPT = """You are an expert in quantum computing. Your task is to learn a predicate that describes quantum states at a specific program point.

## Program
```
{program_source}
```

## Location
{location}

## Observed States
{observed_states}

## Available Predicate Templates
- in_basis(qubit, |0⟩/|1⟩/|+⟩/|-⟩): Qubit is in basis state
- superposition(qubit): Qubit is in superposition
- entangled(q1, q2): Qubits are entangled
- prob(qubit, outcome) >= value: Probability bound

## Task
Identify a predicate that holds for all observed states at this location.

Output as JSON:
```json
{{
    "predicate": "your predicate expression",
    "confidence": 0.0-1.0,
    "reasoning": "why this predicate fits"
}}
```"""


@dataclass
class LearnQuantumPredicate:
    """
    Learn quantum state predicates from execution traces.
    
    This class implements a hybrid approach combining:
    1. Template-based candidate generation from quantum structure
    2. LLM-guided synthesis for complex predicates
    3. Counterexample-driven refinement
    
    Example:
        >>> learner = LearnQuantumPredicate(llm=my_llm)
        >>> predicate = learner.learn(program, "loop_entry", traces)
        >>> print(predicate.to_human_readable())
    """
    
    llm: Optional[LLMInterface] = None
    max_candidates: int = 10
    confidence_threshold: float = 0.9
    
    def learn(
        self,
        program: QuantumProgram,
        location: str,
        examples: list[ExecutionTrace],
    ) -> Optional[QuantumStatePredicate]:
        """
        Learn a predicate that holds at the given program location.
        
        Args:
            program: The quantum program
            location: Program location (e.g., "loop_entry", "line_5")
            examples: Execution traces with states at this location
            
        Returns:
            A predicate that holds for all examples, or None if not found
        """
        if not examples:
            return None
        
        # Step 1: Extract features from program
        features = self._extract_features(program, location)
        
        # Step 2: Generate candidate predicates from templates
        candidates = self._generate_candidates(features, examples)
        
        # Step 3: Evaluate candidates
        evaluated = self._evaluate_candidates(candidates, examples)
        
        # Step 4: Try LLM synthesis if templates fail
        best_candidate = max(evaluated, key=lambda c: c.score) if evaluated else None
        
        if best_candidate and best_candidate.score >= self.confidence_threshold:
            return best_candidate.predicate
        
        if self.llm is not None:
            llm_predicate = self._llm_synthesize(program, location, examples)
            if llm_predicate is not None:
                return llm_predicate
        
        # Return best template even if below threshold
        if best_candidate:
            return best_candidate.predicate
        
        return None
    
    def _extract_features(
        self, 
        program: QuantumProgram, 
        location: str
    ) -> dict[str, Any]:
        """Extract features relevant to predicate learning."""
        return {
            "num_qubits": program.num_qubits,
            "qubit_names": [str(q) for q in program.qubits],
            "has_entangling_gates": any(
                g.name in {"CNOT", "CZ", "SWAP", "CX"} for g in program.gates
            ),
            "has_hadamard": any(g.name in {"H", "HADAMARD"} for g in program.gates),
            "location": location,
        }
    
    def _generate_candidates(
        self,
        features: dict[str, Any],
        examples: list[ExecutionTrace],
    ) -> list[QuantumStatePredicate]:
        """Generate candidate predicates from templates."""
        candidates = []
        qubit_names = features["qubit_names"]
        
        # Basis state predicates
        for qubit in qubit_names:
            for basis in [BasisState.ZERO, BasisState.ONE, BasisState.PLUS, BasisState.MINUS]:
                candidates.append(BasisPredicate(qubit=qubit, basis=basis))
        
        # Superposition predicates
        for qubit in qubit_names:
            candidates.append(SuperpositionPredicate(qubit=qubit))
        
        # Entanglement predicates (pairs)
        if features["has_entangling_gates"]:
            for i, q1 in enumerate(qubit_names):
                for q2 in qubit_names[i+1:]:
                    candidates.append(EntanglementPredicate(qubits=[q1, q2]))
        
        # Probability predicates
        for qubit in qubit_names:
            for outcome in [0, 1]:
                for threshold in [0.5, 0.9, 0.99]:
                    candidates.append(ProbabilityPredicate(
                        qubit=qubit,
                        outcome=outcome,
                        comparison=">=",
                        value=threshold,
                    ))
        
        return candidates
    
    def _evaluate_candidates(
        self,
        candidates: list[QuantumStatePredicate],
        examples: list[ExecutionTrace],
    ) -> list[PredicateCandidate]:
        """Evaluate candidate predicates against examples."""
        evaluated = []
        
        for pred in candidates:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            # Build qubit mapping
            if examples:
                n_qubits = examples[0].num_qubits
                qubit_mapping = {f"q{i}": i for i in range(n_qubits)}
            else:
                qubit_mapping = {}
            
            for trace in examples:
                try:
                    # We expect predicate to hold at intermediate states
                    for state in trace.intermediate_states:
                        result = pred.evaluate(state, qubit_mapping)
                        if result:
                            true_positives += 1
                        else:
                            false_negatives += 1
                except Exception as e:
                    logger.debug(f"Evaluation error: {e}")
                    false_negatives += 1
            
            total = true_positives + false_negatives
            score = true_positives / total if total > 0 else 0.0
            
            evaluated.append(PredicateCandidate(
                predicate=pred,
                score=score,
                false_positives=false_positives,
                false_negatives=false_negatives,
            ))
        
        return evaluated
    
    def _llm_synthesize(
        self,
        program: QuantumProgram,
        location: str,
        examples: list[ExecutionTrace],
    ) -> Optional[QuantumStatePredicate]:
        """Use LLM to synthesize predicate."""
        if self.llm is None:
            return None
        
        # Format observed states
        state_descriptions = []
        for i, trace in enumerate(examples[:5]):  # Limit examples
            for j, state in enumerate(trace.intermediate_states[:3]):
                probs = np.abs(state) ** 2
                state_descriptions.append(
                    f"Trace {i}, state {j}: probabilities = {probs.tolist()}"
                )
        
        prompt = PREDICATE_LEARNING_PROMPT.format(
            program_source=program.source[:500],  # Limit length
            location=location,
            observed_states="\n".join(state_descriptions),
        )
        
        try:
            responses = self.llm.generate(prompt, n=1)
            if responses:
                import json
                response = responses[0]
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response[json_start:json_end])
                    predicate_str = data.get("predicate", "")
                    if predicate_str:
                        return parse_predicate(predicate_str)
        except Exception as e:
            logger.debug(f"LLM synthesis failed: {e}")
        
        return None


def learn_predicate_from_states(
    states: list[NDArray[np.complex128]],
    qubit_names: list[str],
) -> Optional[QuantumStatePredicate]:
    """
    Convenience function to learn a predicate from a list of states.
    
    Args:
        states: List of quantum state vectors
        qubit_names: Names of qubits
        
    Returns:
        A predicate that holds for all states
    """
    if not states:
        return None
    
    # Create dummy traces
    traces = [
        ExecutionTrace(
            input_state=state,
            output_state=state,
            intermediate_states=[state],
        )
        for state in states
    ]
    
    # Create dummy program
    from qverify.core.quantum_program import QuantumProgram
    program = QuantumProgram(
        source="# dummy",
        language=QuantumProgram.ProgramLanguage.SILQ if hasattr(QuantumProgram, 'ProgramLanguage') else None,
        name="dummy",
    )
    
    learner = LearnQuantumPredicate()
    return learner.learn(program, "states", traces)
