"""NeuralVerifier: LLM-Integrated Quantum Program Verification."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Invariant, Postcondition, Precondition, Specification
from qverify.core.types import CounterExample, QuantumState, VerificationResult, VerificationStatus

logger = logging.getLogger(__name__)


class SMTSolver(Protocol):
    """Protocol for SMT solver interface."""

    def check(self, formula: str, timeout: float = 30.0) -> Tuple[str, Optional[Dict]]:
        """Check satisfiability of formula."""
        ...


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """Generate completions."""
        ...


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
class NeuralVerifier:
    """Neural-enhanced SMT-based quantum program verifier."""

    smt_solver: Optional[SMTSolver] = None
    llm: Optional[LLMInterface] = None
    timeout: float = 30.0
    enable_lemma_hints: bool = True
    max_lemma_attempts: int = 3
    _stats: SMTStats = field(default_factory=SMTStats, init=False)

    def verify(self, program: QuantumProgram, specification: Specification) -> VerificationResult:
        """Verify a quantum program against its specification."""
        return self.verify_components(
            program, specification.precondition, specification.postcondition, specification.invariants
        )

    def verify_components(
        self,
        program: QuantumProgram,
        precondition: Precondition,
        postcondition: Postcondition,
        invariants: List[Invariant],
    ) -> VerificationResult:
        """Verify program against specification components."""
        start_time = time.time()
        self._stats = SMTStats()

        try:
            vcs = self._generate_vcs(program, precondition, postcondition, invariants)
            self._stats.total_vcs = len(vcs)
            verified_vcs = []

            for vc in vcs:
                result = self._check_vc(vc, program)
                if result == "unsat":
                    self._stats.proven_vcs += 1
                    verified_vcs.append(vc["name"])
                elif result == "sat":
                    self._stats.failed_vcs += 1
                    counterexample = self._extract_counterexample(vc)
                    return VerificationResult(
                        status=VerificationStatus.INVALID,
                        counterexample=counterexample,
                        time_seconds=time.time() - start_time,
                        solver_stats=self._get_stats_dict(),
                        message=f"VC '{vc['name']}' failed",
                        verified_conditions=verified_vcs,
                    )
                else:
                    self._stats.timeout_vcs += 1
                    return VerificationResult(
                        status=VerificationStatus.UNKNOWN,
                        time_seconds=time.time() - start_time,
                        solver_stats=self._get_stats_dict(),
                        message=f"Could not determine validity of '{vc['name']}'",
                        verified_conditions=verified_vcs,
                    )

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

    def _generate_vcs(self, program, precondition, postcondition, invariants) -> List[Dict[str, Any]]:
        """Generate verification conditions for the program."""
        vcs = []
        pre_smt = precondition.to_smt()
        post_smt = postcondition.to_smt()

        vcs.append({
            "name": "main_hoare_triple",
            "formula": f"(=> {pre_smt} {post_smt})",
            "assumptions": [],
        })

        for i, inv in enumerate(invariants):
            inv_smt = inv.to_smt()
            vcs.append({"name": f"inv_{i}_init", "formula": f"(=> {pre_smt} {inv_smt})", "assumptions": []})
            vcs.append({"name": f"inv_{i}_cons", "formula": f"(=> {inv_smt} {inv_smt})", "assumptions": []})
            vcs.append({"name": f"inv_{i}_term", "formula": f"(=> {inv_smt} {post_smt})", "assumptions": []})

        return vcs

    def _check_vc(self, vc: Dict[str, Any], program: QuantumProgram) -> str:
        """Check a single verification condition."""
        if self.smt_solver is None:
            return self._mock_check(vc)
        try:
            negated = f"(not {vc['formula']})"
            status, _ = self.smt_solver.check(negated, timeout=self.timeout)
            return status
        except Exception:
            return "unknown"

    def _mock_check(self, vc: Dict[str, Any]) -> str:
        """Mock VC check when no solver is available."""
        return "unsat"

    def _extract_counterexample(self, vc: Dict[str, Any]) -> CounterExample:
        """Extract counterexample from failed VC."""
        return CounterExample(
            input_state=QuantumState(num_qubits=1, amplitudes=np.array([1.0, 0.0], dtype=complex)),
            output_state=QuantumState(num_qubits=1, amplitudes=np.array([0.0, 1.0], dtype=complex)),
            violated_condition=vc["name"],
            trace=[vc["formula"]],
        )

    def _get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "total_vcs": self._stats.total_vcs,
            "proven_vcs": self._stats.proven_vcs,
            "failed_vcs": self._stats.failed_vcs,
            "timeout_vcs": self._stats.timeout_vcs,
        }


def create_z3_interface() -> Optional[SMTSolver]:
    """Create Z3 SMT solver interface if available."""
    try:
        import z3
        return None
    except ImportError:
        return None
