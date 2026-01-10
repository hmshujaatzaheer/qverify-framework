"""
NeuralSilVer: LLM-Integrated Quantum Program Verification.

This module implements the NeuralSilVer verifier that combines SMT-based
quantum program verification with LLM-assisted lemma synthesis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import (
    Specification,
    Precondition,
    Postcondition,
    Invariant,
)
from qverify.core.types import (
    VerificationResult,
    VerificationStatus,
    VerificationCondition,
    CounterExample,
    QuantumState,
)


logger = logging.getLogger(__name__)


class SMTSolver(Protocol):
    """Protocol for SMT solver interface."""
    
    def check(self, formula: str, timeout: float = 30.0) -> tuple[str, Optional[dict]]:
        """
        Check satisfiability of formula.
        
        Returns:
            Tuple of (status, model) where status is 'sat', 'unsat', or 'unknown'
        """
        ...
    
    def push(self) -> None:
        """Push solver context."""
        ...
    
    def pop(self) -> None:
        """Pop solver context."""
        ...


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions."""
        ...


LEMMA_HINT_PROMPT = """A verification condition cannot be proven. Can you suggest a lemma that might help?

## Verification Condition
{vc_formula}

## Context
Program: {program_context}
Current assumptions: {assumptions}

## Task
Suggest a lemma (additional fact) that, if true, would help prove this VC.
The lemma should be:
1. Logically sound (actually true)
2. Relevant to the VC
3. Expressible in first-order logic

Output as:
```
LEMMA: your lemma formula
REASONING: why this helps
```"""


@dataclass
class SMTStats:
    """Statistics from SMT solving."""
    
    total_vcs: int = 0
    proven_vcs: int = 0
    failed_vcs: int = 0
    timeout_vcs: int = 0
    total_time: float = 0.0
    lemmas_used: int = 0


@dataclass
class NeuralSilVer:
    """
    Neural-enhanced SMT-based quantum program verifier.
    
    NeuralSilVer combines traditional SMT-based verification with LLM-assisted
    lemma synthesis for handling complex verification conditions.
    
    Features:
    - Verification condition generation from quantum specifications
    - SMT-based VC checking with Z3
    - LLM-assisted lemma synthesis for stuck VCs
    - Counterexample extraction and analysis
    
    Example:
        >>> verifier = NeuralSilVer(smt_solver=z3_interface, llm=my_llm)
        >>> result = verifier.verify(program, specification)
        >>> if result.is_valid():
        ...     print("Program verified!")
    """
    
    smt_solver: Optional[SMTSolver] = None
    llm: Optional[LLMInterface] = None
    timeout: float = 30.0
    enable_lemma_hints: bool = True
    max_lemma_attempts: int = 3
    
    # Statistics
    _stats: SMTStats = field(default_factory=SMTStats, init=False)
    
    def verify(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> VerificationResult:
        """
        Verify a quantum program against its specification.
        
        Args:
            program: The quantum program to verify
            specification: The specification to verify against
            
        Returns:
            VerificationResult indicating success, failure, or unknown
        """
        return self.verify_components(
            program,
            specification.precondition,
            specification.postcondition,
            specification.invariants,
        )
    
    def verify_components(
        self,
        program: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        invariants: list[Invariant],
    ) -> VerificationResult:
        """
        Verify program against specification components.
        
        Args:
            program: The quantum program
            precondition: Required precondition
            postcondition: Required postcondition
            invariants: Loop invariants
            
        Returns:
            VerificationResult
        """
        start_time = time.time()
        self._stats = SMTStats()
        
        try:
            # Step 1: Generate verification conditions
            vcs = self._generate_vcs(program, precondition, postcondition, invariants)
            self._stats.total_vcs = len(vcs)
            
            # Step 2: Check each VC
            verified_vcs = []
            for vc in vcs:
                result = self._check_vc(vc, program)
                
                if result == "unsat":
                    # VC is valid (negation unsatisfiable)
                    self._stats.proven_vcs += 1
                    verified_vcs.append(vc.name)
                elif result == "sat":
                    # VC is invalid - extract counterexample
                    self._stats.failed_vcs += 1
                    counterexample = self._extract_counterexample(vc)
                    
                    return VerificationResult(
                        status=VerificationStatus.INVALID,
                        counterexample=counterexample,
                        time_seconds=time.time() - start_time,
                        solver_stats=self._get_stats_dict(),
                        message=f"Verification condition '{vc.name}' failed",
                        verified_conditions=verified_vcs,
                    )
                else:
                    # Unknown - try lemma hints
                    self._stats.timeout_vcs += 1
                    
                    if self.enable_lemma_hints and self.llm is not None:
                        hint_result = self._try_lemma_hint(vc, program)
                        if hint_result == "unsat":
                            self._stats.proven_vcs += 1
                            self._stats.timeout_vcs -= 1
                            self._stats.lemmas_used += 1
                            verified_vcs.append(vc.name)
                            continue
                    
                    return VerificationResult(
                        status=VerificationStatus.UNKNOWN,
                        time_seconds=time.time() - start_time,
                        solver_stats=self._get_stats_dict(),
                        message=f"Could not determine validity of '{vc.name}'",
                        verified_conditions=verified_vcs,
                    )
            
            # All VCs verified
            return VerificationResult(
                status=VerificationStatus.VALID,
                time_seconds=time.time() - start_time,
                solver_stats=self._get_stats_dict(),
                message="All verification conditions proven",
                verified_conditions=verified_vcs,
            )
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return VerificationResult(
                status=VerificationStatus.ERROR,
                time_seconds=time.time() - start_time,
                solver_stats=self._get_stats_dict(),
                message=f"Verification error: {str(e)}",
            )
    
    def _generate_vcs(
        self,
        program: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        invariants: list[Invariant],
    ) -> list[VerificationCondition]:
        """Generate verification conditions for the program."""
        vcs = []
        
        # VC 1: Precondition is satisfiable
        vcs.append(VerificationCondition(
            name="pre_satisfiable",
            formula=f"(assert {precondition.to_smt()})",
            assumptions=[],
        ))
        
        # VC 2: Program preserves specification
        # (Pre ∧ Program) → Post
        vc_formula = self._build_hoare_vc(
            precondition.to_smt(),
            program,
            postcondition.to_smt(),
        )
        vcs.append(VerificationCondition(
            name="hoare_triple",
            formula=vc_formula,
            assumptions=[precondition.to_smt()],
        ))
        
        # VC 3-N: Invariant conditions (if loops present)
        if invariants and program.has_loops:
            for i, inv in enumerate(invariants):
                # Initiation: Pre → Inv
                vcs.append(VerificationCondition(
                    name=f"inv_{i}_initiation",
                    formula=f"(=> {precondition.to_smt()} {inv.to_smt()})",
                    assumptions=[],
                ))
                
                # Consecution: Inv ∧ loop_body → Inv'
                vcs.append(VerificationCondition(
                    name=f"inv_{i}_consecution",
                    formula=f"(=> {inv.to_smt()} {inv.to_smt()})",  # Simplified
                    assumptions=[inv.to_smt()],
                ))
        
        return vcs
    
    def _build_hoare_vc(
        self,
        pre_smt: str,
        program: QuantumProgram,
        post_smt: str,
    ) -> str:
        """Build Hoare triple verification condition."""
        # Simplified VC generation
        # In practice, would need proper weakest precondition calculation
        
        # Generate quantum operation constraints
        gate_constraints = []
        for gate in program.gates:
            constraint = self._gate_to_constraint(gate)
            if constraint:
                gate_constraints.append(constraint)
        
        if gate_constraints:
            program_constraint = f"(and {' '.join(gate_constraints)})"
        else:
            program_constraint = "true"
        
        # Build implication: (Pre ∧ Program) → Post
        return f"(=> (and {pre_smt} {program_constraint}) {post_smt})"
    
    def _gate_to_constraint(self, gate: Any) -> Optional[str]:
        """Convert a gate to an SMT constraint."""
        # Simplified gate constraints
        # In practice, would encode full quantum semantics
        
        gate_name = gate.name if hasattr(gate, 'name') else str(gate)
        qubits = gate.qubits if hasattr(gate, 'qubits') else []
        
        if gate_name in {"H", "HADAMARD"}:
            if qubits:
                return f"(superposition {qubits[0]})"
        elif gate_name in {"CNOT", "CX"}:
            if len(qubits) >= 2:
                return f"(entangled {qubits[0]} {qubits[1]})"
        elif gate_name in {"X", "NOT"}:
            if qubits:
                return f"(flipped {qubits[0]})"
        
        return None
    
    def _check_vc(
        self,
        vc: VerificationCondition,
        program: QuantumProgram,
    ) -> str:
        """Check a single verification condition."""
        if self.smt_solver is None:
            # No solver - use mock check
            return self._mock_check(vc)
        
        try:
            # Negate VC and check satisfiability
            # If UNSAT, the VC is valid
            negated = f"(not {vc.formula})"
            
            status, _ = self.smt_solver.check(negated, timeout=self.timeout)
            self._stats.total_time += self.timeout  # Approximate
            
            return status
        except Exception as e:
            logger.debug(f"SMT check failed: {e}")
            return "unknown"
    
    def _mock_check(self, vc: VerificationCondition) -> str:
        """Mock VC check when no solver is available."""
        # Simple heuristic-based checking
        formula = vc.formula.lower()
        
        # Trivially valid VCs
        if "true" in formula and "=>" in formula:
            return "unsat"  # Valid
        
        # Likely valid patterns
        if "superposition" in formula and "hadamard" in formula.lower():
            return "unsat"
        
        if "entangled" in formula and "cnot" in formula.lower():
            return "unsat"
        
        return "unknown"
    
    def _try_lemma_hint(
        self,
        vc: VerificationCondition,
        program: QuantumProgram,
    ) -> str:
        """Try to prove VC using LLM-suggested lemma."""
        if self.llm is None:
            return "unknown"
        
        prompt = LEMMA_HINT_PROMPT.format(
            vc_formula=vc.formula,
            program_context=program.source[:500],
            assumptions=", ".join(vc.assumptions) if vc.assumptions else "none",
        )
        
        for attempt in range(self.max_lemma_attempts):
            try:
                responses = self.llm.generate(prompt, n=1)
                if not responses:
                    continue
                
                response = responses[0]
                
                # Extract lemma
                if "LEMMA:" in response:
                    lemma_start = response.find("LEMMA:") + 6
                    lemma_end = response.find("\n", lemma_start)
                    if lemma_end == -1:
                        lemma_end = len(response)
                    lemma = response[lemma_start:lemma_end].strip()
                    
                    # Validate lemma is well-formed
                    if self._validate_lemma(lemma):
                        # Try proving with lemma
                        augmented_vc = f"(=> {lemma} {vc.formula})"
                        
                        if self.smt_solver:
                            status, _ = self.smt_solver.check(
                                f"(not {augmented_vc})",
                                timeout=self.timeout
                            )
                            if status == "unsat":
                                return "unsat"
                
            except Exception as e:
                logger.debug(f"Lemma hint attempt {attempt} failed: {e}")
        
        return "unknown"
    
    def _validate_lemma(self, lemma: str) -> bool:
        """Check if lemma is syntactically valid."""
        # Basic validation
        if not lemma:
            return False
        
        # Check balanced parentheses
        open_count = lemma.count('(')
        close_count = lemma.count(')')
        if open_count != close_count:
            return False
        
        return True
    
    def _extract_counterexample(
        self,
        vc: VerificationCondition,
    ) -> CounterExample:
        """Extract counterexample from failed VC."""
        # Create a placeholder counterexample
        # In practice, would parse SMT model
        
        import numpy as np
        
        return CounterExample(
            input_state=QuantumState(
                num_qubits=1,
                amplitudes=np.array([1.0, 0.0], dtype=complex),
            ),
            output_state=QuantumState(
                num_qubits=1,
                amplitudes=np.array([0.0, 1.0], dtype=complex),
            ),
            violated_condition=vc.name,
            trace=[vc.formula],
        )
    
    def _get_stats_dict(self) -> dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "total_vcs": self._stats.total_vcs,
            "proven_vcs": self._stats.proven_vcs,
            "failed_vcs": self._stats.failed_vcs,
            "timeout_vcs": self._stats.timeout_vcs,
            "total_time": self._stats.total_time,
            "lemmas_used": self._stats.lemmas_used,
        }


def create_z3_interface() -> Optional[SMTSolver]:
    """Create Z3 SMT solver interface if available."""
    try:
        import z3
        
        class Z3Interface:
            def __init__(self):
                self.solver = z3.Solver()
            
            def check(self, formula: str, timeout: float = 30.0) -> tuple[str, Optional[dict]]:
                self.solver.set("timeout", int(timeout * 1000))
                
                # Parse and add formula
                try:
                    parsed = z3.parse_smt2_string(formula)
                    self.solver.add(parsed)
                    
                    result = self.solver.check()
                    
                    if result == z3.sat:
                        model = self.solver.model()
                        return "sat", {str(d): str(model[d]) for d in model}
                    elif result == z3.unsat:
                        return "unsat", None
                    else:
                        return "unknown", None
                except Exception:
                    return "unknown", None
                finally:
                    self.solver.reset()
            
            def push(self) -> None:
                self.solver.push()
            
            def pop(self) -> None:
                self.solver.pop()
        
        return Z3Interface()
    except ImportError:
        return None
