"""
SMT Solver Interface for QVERIFY.

This module provides a unified interface for interacting with SMT solvers,
with primary support for Z3 and extensibility for other solvers.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SolverResult(Enum):
    """Result of an SMT query."""
    
    SAT = "sat"
    """Formula is satisfiable."""
    
    UNSAT = "unsat"
    """Formula is unsatisfiable."""
    
    UNKNOWN = "unknown"
    """Solver could not determine satisfiability."""
    
    TIMEOUT = "timeout"
    """Solver timed out."""
    
    ERROR = "error"
    """An error occurred."""


@dataclass
class SolverStats:
    """Statistics from SMT solving."""
    
    num_queries: int = 0
    total_time: float = 0.0
    sat_count: int = 0
    unsat_count: int = 0
    unknown_count: int = 0
    timeout_count: int = 0


@dataclass
class QueryResult:
    """Result of an SMT query."""
    
    status: SolverResult
    model: Optional[dict[str, Any]] = None
    time_seconds: float = 0.0
    statistics: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_sat(self) -> bool:
        return self.status == SolverResult.SAT
    
    @property
    def is_unsat(self) -> bool:
        return self.status == SolverResult.UNSAT


class SMTSolver(ABC):
    """Abstract base class for SMT solvers."""
    
    @abstractmethod
    def check(self, formula: str, timeout: float = 30.0) -> QueryResult:
        """Check satisfiability of formula."""
        pass
    
    @abstractmethod
    def push(self) -> None:
        """Push solver context."""
        pass
    
    @abstractmethod
    def pop(self) -> None:
        """Pop solver context."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset solver state."""
        pass
    
    @abstractmethod
    def add(self, formula: str) -> None:
        """Add formula to solver context."""
        pass


class Z3Solver(SMTSolver):
    """Z3 SMT solver interface."""
    
    def __init__(self) -> None:
        self._solver = None
        self._z3 = None
        self._stats = SolverStats()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Z3 solver."""
        try:
            import z3
            self._z3 = z3
            self._solver = z3.Solver()
            logger.info("Z3 solver initialized successfully")
        except ImportError:
            logger.warning("Z3 not available. Install with: pip install z3-solver")
            raise ImportError("Z3 solver required but not installed")
    
    def check(self, formula: str, timeout: float = 30.0) -> QueryResult:
        """Check satisfiability using Z3."""
        start_time = time.time()
        self._stats.num_queries += 1
        
        try:
            # Set timeout
            self._solver.set("timeout", int(timeout * 1000))
            
            # Parse and add formula
            try:
                parsed = self._z3.parse_smt2_string(formula)
                self._solver.push()
                self._solver.add(parsed)
            except Exception as e:
                logger.error(f"Failed to parse formula: {e}")
                return QueryResult(
                    status=SolverResult.ERROR,
                    time_seconds=time.time() - start_time,
                    statistics={"error": str(e)},
                )
            
            # Check satisfiability
            result = self._solver.check()
            elapsed = time.time() - start_time
            
            if result == self._z3.sat:
                self._stats.sat_count += 1
                model = self._solver.model()
                model_dict = {str(d): str(model[d]) for d in model}
                
                self._solver.pop()
                return QueryResult(
                    status=SolverResult.SAT,
                    model=model_dict,
                    time_seconds=elapsed,
                )
            
            elif result == self._z3.unsat:
                self._stats.unsat_count += 1
                self._solver.pop()
                return QueryResult(
                    status=SolverResult.UNSAT,
                    time_seconds=elapsed,
                )
            
            else:
                self._stats.unknown_count += 1
                self._solver.pop()
                return QueryResult(
                    status=SolverResult.UNKNOWN,
                    time_seconds=elapsed,
                )
                
        except Exception as e:
            logger.error(f"Z3 error: {e}")
            return QueryResult(
                status=SolverResult.ERROR,
                time_seconds=time.time() - start_time,
                statistics={"error": str(e)},
            )
        finally:
            self._stats.total_time += time.time() - start_time
    
    def push(self) -> None:
        """Push Z3 context."""
        self._solver.push()
    
    def pop(self) -> None:
        """Pop Z3 context."""
        self._solver.pop()
    
    def reset(self) -> None:
        """Reset Z3 solver."""
        self._solver.reset()
    
    def add(self, formula: str) -> None:
        """Add formula to Z3."""
        try:
            parsed = self._z3.parse_smt2_string(formula)
            self._solver.add(parsed)
        except Exception as e:
            logger.error(f"Failed to add formula: {e}")
    
    def get_stats(self) -> SolverStats:
        """Get solver statistics."""
        return self._stats


class MockSolver(SMTSolver):
    """Mock SMT solver for testing."""
    
    def __init__(self, default_result: SolverResult = SolverResult.UNSAT) -> None:
        self.default_result = default_result
        self._context_stack: list[list[str]] = [[]]
        self._stats = SolverStats()
    
    def check(self, formula: str, timeout: float = 30.0) -> QueryResult:
        """Return mock result."""
        self._stats.num_queries += 1
        
        # Simple heuristics for mock results
        formula_lower = formula.lower()
        
        if "false" in formula_lower:
            return QueryResult(status=SolverResult.UNSAT, time_seconds=0.01)
        
        if "true" in formula_lower and "=>" in formula_lower:
            return QueryResult(status=SolverResult.UNSAT, time_seconds=0.01)
        
        return QueryResult(status=self.default_result, time_seconds=0.01)
    
    def push(self) -> None:
        """Push mock context."""
        self._context_stack.append([])
    
    def pop(self) -> None:
        """Pop mock context."""
        if len(self._context_stack) > 1:
            self._context_stack.pop()
    
    def reset(self) -> None:
        """Reset mock solver."""
        self._context_stack = [[]]
    
    def add(self, formula: str) -> None:
        """Add formula to mock context."""
        self._context_stack[-1].append(formula)


def create_solver(solver_type: str = "z3") -> SMTSolver:
    """
    Factory function to create SMT solver.
    
    Args:
        solver_type: Type of solver ("z3", "mock")
        
    Returns:
        SMTSolver instance
    """
    if solver_type.lower() == "z3":
        try:
            return Z3Solver()
        except ImportError:
            logger.warning("Z3 not available, falling back to mock solver")
            return MockSolver()
    
    elif solver_type.lower() == "mock":
        return MockSolver()
    
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


# Quantum-specific SMT theories
QUANTUM_SMT_PRELUDE = """
; Quantum state theory for QVERIFY
(declare-sort Qubit 0)
(declare-sort QuantumState 0)
(declare-sort Basis 0)

; Basis states
(declare-const zero Basis)
(declare-const one Basis)
(declare-const plus Basis)
(declare-const minus Basis)

; State predicates
(declare-fun in_basis (Qubit Basis) Bool)
(declare-fun superposition (Qubit) Bool)
(declare-fun entangled (Qubit Qubit) Bool)
(declare-fun prob (Qubit Basis) Real)
(declare-fun amplitude (Qubit Basis) Real)

; Gate effects (simplified)
(declare-fun hadamard_wp (Qubit Bool) Bool)
(declare-fun pauli_x_wp (Qubit Bool) Bool)
(declare-fun cnot_wp (Qubit Qubit Bool) Bool)

; Axioms
(assert (forall ((q Qubit)) 
    (=> (in_basis q zero) (not (in_basis q one)))))
(assert (forall ((q Qubit))
    (=> (superposition q) (and (> (prob q zero) 0) (> (prob q one) 0)))))
"""
