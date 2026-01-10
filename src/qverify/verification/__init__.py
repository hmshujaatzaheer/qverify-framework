"""
Verification module for QVERIFY.

This module provides the verification infrastructure including the NeuralSilVer
verifier, VC generation, SMT interface, and counterexample analysis.
"""

from qverify.verification.neural_silver import (
    NeuralSilVer,
    SMTStats,
    create_z3_interface,
)
from qverify.verification.vc_generator import (
    VCGenerator,
    VCGeneratorConfig,
    generate_vcs,
)
from qverify.verification.smt_interface import (
    SMTSolver,
    Z3Solver,
    MockSolver,
    SolverResult,
    QueryResult,
    SolverStats,
    create_solver,
    QUANTUM_SMT_PRELUDE,
)
from qverify.verification.counterexample import (
    CounterexampleAnalyzer,
    CounterexampleDiagnosis,
    ViolationType,
    create_counterexample_from_model,
    format_counterexample,
)

__all__ = [
    # NeuralSilVer
    "NeuralSilVer",
    "SMTStats",
    "create_z3_interface",
    # VC Generator
    "VCGenerator",
    "VCGeneratorConfig",
    "generate_vcs",
    # SMT Interface
    "SMTSolver",
    "Z3Solver",
    "MockSolver",
    "SolverResult",
    "QueryResult",
    "SolverStats",
    "create_solver",
    "QUANTUM_SMT_PRELUDE",
    # Counterexample
    "CounterexampleAnalyzer",
    "CounterexampleDiagnosis",
    "ViolationType",
    "create_counterexample_from_model",
    "format_counterexample",
]
