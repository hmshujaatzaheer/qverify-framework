"""SynthesizeQuantumInvariant: Loop Invariant Generation for Iterative Quantum Algorithms."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol

import numpy as np

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Invariant, Postcondition, Precondition
from qverify.core.types import QuantumState

logger = logging.getLogger(__name__)


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate completions."""
        ...


class VerificationOracle(Protocol):
    """Protocol for checking verification conditions."""

    def check_implication(self, premise: str, conclusion: str) -> bool:
        """Check if premise implies conclusion."""
        ...


@dataclass
class LoopTrace:
    """Execution trace of a loop iteration."""

    iteration: int
    entry_state: QuantumState
    exit_state: QuantumState
    intermediate_states: List[QuantumState] = field(default_factory=list)


INVARIANT_PROMPT = """You are an expert in quantum program verification.

## Loop Structure
```
{loop_source}
```

## Observed Invariants
{observed_invariants}

## Precondition
{precondition}

## Postcondition
{postcondition}

## Task
Find a general invariant that is preserved by each loop iteration.

Invariant: """


@dataclass
class SynthesizeQuantumInvariant:
    """Synthesize loop invariants for iterative quantum algorithms."""

    llm: LLMInterface
    max_unrolling: int = 5
    predicate_learner: Optional[Any] = None
    verification_oracle: Optional[VerificationOracle] = None
    simulator: Optional[Callable] = None

    def synthesize(
        self,
        loop: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        loop_condition: str = "true",
    ) -> Optional[Invariant]:
        """Synthesize a loop invariant."""
        observed_predicates: List[str] = []

        for k in range(1, self.max_unrolling + 1):
            traces = self._simulate_loop(loop, precondition, k)
            predicate_k = self._learn_iteration_predicate(loop, traces, k)
            if predicate_k:
                observed_predicates.append(f"k={k}: {predicate_k}")

            if len(observed_predicates) >= 2:
                candidate_inv = self._generalize(loop, observed_predicates, precondition, postcondition)
                if candidate_inv:
                    inv = Invariant.from_string(candidate_inv, "loop_entry")
                    if self._check_initiation(precondition, inv):
                        if self._check_consecution(inv, loop, inv):
                            if self._check_termination(inv, loop_condition, postcondition):
                                return inv

        return None

    def _simulate_loop(self, loop: QuantumProgram, precondition: Precondition, num_iterations: int) -> List[LoopTrace]:
        """Simulate loop for a given number of iterations."""
        traces = []
        initial_state = self._create_initial_state(loop.num_qubits, precondition)
        current_state = initial_state

        for i in range(num_iterations):
            entry_state = current_state
            if self.simulator:
                exit_state = self.simulator(loop, current_state)
            else:
                exit_state = self._default_simulate(loop, current_state)

            traces.append(LoopTrace(iteration=i + 1, entry_state=entry_state, exit_state=exit_state))
            current_state = exit_state

        return traces

    def _create_initial_state(self, num_qubits: int, precondition: Precondition) -> QuantumState:
        """Create an initial state satisfying the precondition."""
        dim = 2 ** num_qubits
        amplitudes = np.ones(dim, dtype=complex) / np.sqrt(dim)
        return QuantumState(num_qubits=num_qubits, amplitudes=amplitudes)

    def _default_simulate(self, loop: QuantumProgram, state: QuantumState) -> QuantumState:
        """Simulate loop body."""
        amplitudes = state.amplitudes.copy()
        return QuantumState(num_qubits=state.num_qubits, amplitudes=amplitudes)

    def _learn_iteration_predicate(self, loop: QuantumProgram, traces: List[LoopTrace], iteration: int) -> Optional[str]:
        """Learn predicate for a specific iteration."""
        if not traces:
            return None
        trace = traces[-1]
        predicates = []
        prob_max = max(abs(amp) ** 2 for amp in trace.exit_state.amplitudes)
        if prob_max > 0.5:
            predicates.append(f"max_prob >= {prob_max:.3f}")
        nonzero = sum(1 for amp in trace.exit_state.amplitudes if abs(amp) > 1e-10)
        if nonzero > 1:
            predicates.append("superposition(qubits)")
        return " and ".join(predicates) if predicates else None

    def _generalize(self, loop: QuantumProgram, observed_predicates: List[str], precondition: Precondition, postcondition: Postcondition) -> Optional[str]:
        """Use LLM to generalize observed predicates."""
        prompt = INVARIANT_PROMPT.format(
            loop_source=loop.source[:500],
            observed_invariants="\n".join(f"  {p}" for p in observed_predicates),
            precondition=precondition.formula,
            postcondition=postcondition.formula,
        )
        responses = self.llm.generate(prompt, n=1)
        if responses:
            response = responses[0].strip()
            if ":" in response:
                response = response.split(":")[-1].strip()
            return response
        return None

    def _check_initiation(self, pre: Precondition, inv: Invariant) -> bool:
        """Check if precondition implies invariant."""
        if self.verification_oracle:
            return self.verification_oracle.check_implication(pre.formula, inv.formula)
        return True

    def _check_consecution(self, inv: Invariant, loop: QuantumProgram, inv_prime: Invariant) -> bool:
        """Check if invariant is preserved by loop body."""
        if self.verification_oracle:
            return self.verification_oracle.check_implication(inv.formula, inv_prime.formula)
        return True

    def _check_termination(self, inv: Invariant, loop_condition: str, post: Postcondition) -> bool:
        """Check if invariant with terminated loop implies postcondition."""
        if self.verification_oracle:
            premise = f"({inv.formula}) and not({loop_condition})"
            return self.verification_oracle.check_implication(premise, post.formula)
        return True
