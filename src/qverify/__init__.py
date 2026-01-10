"""
QVERIFY: LLM-Assisted Formal Verification of Quantum Programs
==============================================================

QVERIFY is a framework that integrates Large Language Model (LLM) reasoning
capabilities with formal verification of quantum programs.

Key Components:
    - QuantumSpecSynth: LLM-guided specification synthesis
    - NeuralSilVer: Integrated verification with SMT solving
    - QVerifyBench: Benchmark for quantum program verification

Basic Usage:
    >>> from qverify import QVerify, QuantumProgram
    >>> qv = QVerify(llm="claude-3.5-sonnet")
    >>> program = QuantumProgram.from_silq("def hadamard_measure() {...}")
    >>> spec = qv.synthesize_specification(program)
    >>> result = qv.verify(program, spec)
    >>> print(result.status)
    'Valid'

For more information, see: https://github.com/hmshujaatzaheer/qverify-framework
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import (
    Specification,
    Precondition,
    Postcondition,
    Invariant,
)
from qverify.core.quantum_state import (
    QuantumStatePredicate,
    BasisPredicate,
    EntanglementPredicate,
    AmplitudePredicate,
    ProbabilityPredicate,
)
from qverify.core.types import (
    VerificationResult,
    VerificationStatus,
    CounterExample,
    SynthesisResult,
)
from qverify.algorithms.spec_synth import QuantumSpecSynth
from qverify.algorithms.predicate_learning import LearnQuantumPredicate
from qverify.algorithms.invariant_synth import SynthesizeQuantumInvariant
from qverify.algorithms.spec_repair import RepairSpecification
from qverify.verification.neural_silver import NeuralSilVer
from qverify.qverify import QVerify

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Main class
    "QVerify",
    # Core types
    "QuantumProgram",
    "Specification",
    "Precondition",
    "Postcondition",
    "Invariant",
    # Quantum state predicates
    "QuantumStatePredicate",
    "BasisPredicate",
    "EntanglementPredicate",
    "AmplitudePredicate",
    "ProbabilityPredicate",
    # Result types
    "VerificationResult",
    "VerificationStatus",
    "CounterExample",
    "SynthesisResult",
    # Algorithms
    "QuantumSpecSynth",
    "LearnQuantumPredicate",
    "SynthesizeQuantumInvariant",
    "RepairSpecification",
    # Verification
    "NeuralSilVer",
]
