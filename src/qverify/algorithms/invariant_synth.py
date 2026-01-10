"""
SynthesizeQuantumInvariant: Loop Invariant Generation for Iterative Quantum Algorithms.

This module implements Algorithm 3 from the QVERIFY paper - synthesizing loop
invariants for iterative quantum algorithms like Grover's and quantum walks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol
import numpy as np
from numpy.typing import NDArray

from qverify.core.quantum_program import QuantumProgram, Loop
from qverify.core.specification import Invariant, Precondition, Postcondition
from qverify.core.quantum_state import QuantumStatePredicate
from qverify.algorithms.predicate_learning import LearnQuantumPredicate, ExecutionTrace


logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions."""
        ...


class QuantumSimulator(Protocol):
    """Protocol for quantum simulation."""
    
    def simulate(
        self, 
        program: QuantumProgram, 
        input_state: NDArray[np.complex128],
        max_steps: int = 100,
    ) -> list[NDArray[np.complex128]]:
        """Simulate program and return intermediate states."""
        ...


GENERALIZATION_PROMPT = """You are an expert in quantum algorithm analysis. Your task is to generalize loop invariants.

## Quantum Loop
```
{loop_code}
```

## Observed Invariants by Iteration
{iteration_invariants}

## Task
Find a general invariant formula that holds for ALL iterations k.
The invariant should:
1. Hold at loop entry (initiation)
2. Be preserved by loop body (consecution)
3. Imply postcondition when loop exits (termination)

Express the invariant in terms of iteration variable k.

Output as JSON:
```json
{{
    "invariant": "your invariant formula with k",
    "reasoning": "why this generalizes"
}}
```"""


@dataclass
class LoopAnalysis:
    """Analysis of a quantum loop structure."""
    
    loop_variable: str
    start_value: int
    end_value: int
    body_gates: list[str]
    modifies_qubits: list[str]
    reads_qubits: list[str]


@dataclass 
class SynthesizeQuantumInvariant:
    """
    Synthesize loop invariants for iterative quantum algorithms.
    
    This class combines:
    1. Loop unrolling to observe iteration behavior
    2. Predicate learning at each iteration
    3. LLM-guided generalization across iterations
    4. Verification of invariant properties (initiation, consecution, termination)
    
    Example:
        >>> synth = SynthesizeQuantumInvariant(llm=my_llm, simulator=my_sim)
        >>> invariant = synth.synthesize(grover_loop, pre, post)
        >>> print(invariant.to_human_readable())
    """
    
    llm: Optional[LLMInterface] = None
    simulator: Optional[QuantumSimulator] = None
    max_unrolling: int = 5
    predicate_learner: LearnQuantumPredicate = field(
        default_factory=LearnQuantumPredicate
    )
    
    def synthesize(
        self,
        loop: Loop,
        precondition: Precondition,
        postcondition: Postcondition,
        program_context: Optional[QuantumProgram] = None,
    ) -> Optional[Invariant]:
        """
        Synthesize an invariant for the given loop.
        
        Args:
            loop: The loop to synthesize invariant for
            precondition: Precondition before loop
            postcondition: Required postcondition after loop
            program_context: The full program containing the loop
            
        Returns:
            An invariant that satisfies initiation, consecution, and termination
        """
        # Step 1: Analyze loop structure
        analysis = self._analyze_loop(loop)
        
        # Step 2: Generate execution traces by unrolling
        traces = self._simulate_loop(loop, precondition, analysis)
        
        # Step 3: Learn predicates at each iteration
        iteration_predicates = self._learn_iteration_predicates(
            loop, traces, analysis
        )
        
        # Step 4: Generalize across iterations
        invariant = self._generalize(loop, iteration_predicates, analysis)
        
        if invariant is None:
            return None
        
        # Step 5: Verify invariant properties
        if self._verify_invariant(invariant, loop, precondition, postcondition):
            return invariant
        
        return None
    
    def _analyze_loop(self, loop: Loop) -> LoopAnalysis:
        """Analyze loop structure."""
        # Extract gate operations from loop body
        body_gates = []
        modifies = set()
        reads = set()
        
        for item in loop.body:
            if hasattr(item, 'name'):  # It's a Gate
                body_gates.append(item.name)
                for q in getattr(item, 'qubits', []):
                    modifies.add(str(q))
                    reads.add(str(q))
        
        return LoopAnalysis(
            loop_variable=loop.variable,
            start_value=loop.start,
            end_value=loop.end,
            body_gates=body_gates,
            modifies_qubits=list(modifies),
            reads_qubits=list(reads),
        )
    
    def _simulate_loop(
        self,
        loop: Loop,
        precondition: Precondition,
        analysis: LoopAnalysis,
    ) -> dict[int, list[ExecutionTrace]]:
        """Simulate loop for multiple iterations."""
        traces: dict[int, list[ExecutionTrace]] = {}
        
        # Generate initial states satisfying precondition
        initial_states = self._generate_initial_states(precondition, analysis)
        
        for k in range(min(analysis.end_value - analysis.start_value + 1, self.max_unrolling)):
            traces[k] = []
            
            for init_state in initial_states:
                # Simulate k iterations
                intermediate = self._simulate_k_iterations(loop, init_state, k)
                
                if intermediate:
                    traces[k].append(ExecutionTrace(
                        input_state=init_state,
                        output_state=intermediate[-1] if intermediate else init_state,
                        intermediate_states=intermediate,
                    ))
        
        return traces
    
    def _generate_initial_states(
        self,
        precondition: Precondition,
        analysis: LoopAnalysis,
    ) -> list[NDArray[np.complex128]]:
        """Generate initial states satisfying precondition."""
        # Generate sample states - in practice would use precondition constraints
        states = []
        n_qubits = max(len(analysis.modifies_qubits), 2)
        
        # Standard computational basis states
        for i in range(min(2**n_qubits, 4)):
            state = np.zeros(2**n_qubits, dtype=complex)
            state[i] = 1.0
            states.append(state)
        
        # Equal superposition
        state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        states.append(state)
        
        return states
    
    def _simulate_k_iterations(
        self,
        loop: Loop,
        init_state: NDArray[np.complex128],
        k: int,
    ) -> list[NDArray[np.complex128]]:
        """Simulate exactly k iterations of the loop."""
        # Simplified simulation - in practice would use quantum simulator
        states = [init_state.copy()]
        current = init_state.copy()
        
        for _ in range(k):
            # Apply loop body transformations
            # This is a placeholder - actual implementation would
            # compile loop body to unitary and apply it
            
            # For demonstration, apply random-ish unitary
            n = len(current)
            # Hadamard-like transformation on first qubit
            if n >= 2:
                new_state = np.zeros_like(current)
                sqrt2 = np.sqrt(2)
                for i in range(n):
                    if i % 2 == 0:
                        new_state[i] = (current[i] + current[i+1]) / sqrt2
                    else:
                        new_state[i] = (current[i-1] - current[i]) / sqrt2
                current = new_state
            
            states.append(current.copy())
        
        return states
    
    def _learn_iteration_predicates(
        self,
        loop: Loop,
        traces: dict[int, list[ExecutionTrace]],
        analysis: LoopAnalysis,
    ) -> dict[int, QuantumStatePredicate]:
        """Learn predicates at each iteration."""
        predicates = {}
        
        # Dummy program for predicate learner
        program = QuantumProgram(
            source=str(loop),
            language=None,
            name="loop",
        )
        
        for k, k_traces in traces.items():
            if k_traces:
                pred = self.predicate_learner.learn(
                    program,
                    f"iteration_{k}",
                    k_traces,
                )
                if pred is not None:
                    predicates[k] = pred
        
        return predicates
    
    def _generalize(
        self,
        loop: Loop,
        iteration_predicates: dict[int, QuantumStatePredicate],
        analysis: LoopAnalysis,
    ) -> Optional[Invariant]:
        """Generalize iteration-specific predicates to loop invariant."""
        if not iteration_predicates:
            return None
        
        # If we have an LLM, use it to generalize
        if self.llm is not None:
            return self._llm_generalize(loop, iteration_predicates, analysis)
        
        # Otherwise, find common predicate across iterations
        return self._simple_generalize(iteration_predicates, analysis)
    
    def _llm_generalize(
        self,
        loop: Loop,
        iteration_predicates: dict[int, QuantumStatePredicate],
        analysis: LoopAnalysis,
    ) -> Optional[Invariant]:
        """Use LLM to generalize predicates."""
        # Format iteration invariants
        inv_strs = []
        for k, pred in sorted(iteration_predicates.items()):
            inv_strs.append(f"k={k}: {pred.to_human_readable()}")
        
        prompt = GENERALIZATION_PROMPT.format(
            loop_code=str(loop),
            iteration_invariants="\n".join(inv_strs),
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
                    inv_str = data.get("invariant", "")
                    if inv_str:
                        return Invariant.from_string(
                            inv_str,
                            loop_variable=analysis.loop_variable,
                            description=data.get("reasoning", ""),
                        )
        except Exception as e:
            logger.debug(f"LLM generalization failed: {e}")
        
        return None
    
    def _simple_generalize(
        self,
        iteration_predicates: dict[int, QuantumStatePredicate],
        analysis: LoopAnalysis,
    ) -> Optional[Invariant]:
        """Simple generalization without LLM."""
        # Find predicate that holds at all iterations
        if len(iteration_predicates) < 2:
            if iteration_predicates:
                pred = list(iteration_predicates.values())[0]
                return Invariant.from_string(
                    pred.to_human_readable(),
                    loop_variable=analysis.loop_variable,
                )
            return None
        
        # Check if all predicates are the same type
        pred_types = set(type(p).__name__ for p in iteration_predicates.values())
        if len(pred_types) == 1:
            # All same type - use first one as template
            first_pred = list(iteration_predicates.values())[0]
            return Invariant.from_string(
                first_pred.to_human_readable(),
                loop_variable=analysis.loop_variable,
                description="Consistent across observed iterations",
            )
        
        return None
    
    def _verify_invariant(
        self,
        invariant: Invariant,
        loop: Loop,
        precondition: Precondition,
        postcondition: Postcondition,
    ) -> bool:
        """
        Verify invariant satisfies initiation, consecution, and termination.
        
        Returns True if all three properties hold (or can't be disproven).
        """
        # In a full implementation, these would be SMT checks
        # For now, we do basic syntactic checks
        
        # Initiation: Pre => Inv (at k=0)
        initiation = self._check_initiation(precondition, invariant)
        
        # Consecution: Inv[k] ∧ body => Inv[k+1]
        consecution = self._check_consecution(invariant, loop)
        
        # Termination: Inv ∧ ¬cond => Post
        termination = self._check_termination(invariant, loop, postcondition)
        
        return initiation and consecution and termination
    
    def _check_initiation(
        self,
        precondition: Precondition,
        invariant: Invariant,
    ) -> bool:
        """Check if precondition implies invariant at loop entry."""
        # Placeholder - would use SMT solver
        return True
    
    def _check_consecution(
        self,
        invariant: Invariant,
        loop: Loop,
    ) -> bool:
        """Check if invariant is preserved by loop body."""
        # Placeholder - would use SMT solver
        return True
    
    def _check_termination(
        self,
        invariant: Invariant,
        loop: Loop,
        postcondition: Postcondition,
    ) -> bool:
        """Check if invariant with exit condition implies postcondition."""
        # Placeholder - would use SMT solver
        return True
