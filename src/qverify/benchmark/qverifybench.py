"""
QVerifyBench: Benchmark for Quantum Program Verification.

This module provides a benchmark framework for evaluating LLMs on quantum
program verification tasks across five difficulty tiers.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Specification
from qverify.core.types import SynthesisStatus, VerificationStatus


logger = logging.getLogger(__name__)


class BenchmarkTier(Enum):
    """Difficulty tiers for QVerifyBench."""
    
    T1 = "T1"  # Basic: Single qubit, no loops
    T2 = "T2"  # Intermediate: Multi-qubit, entanglement
    T3 = "T3"  # Standard: Loops, oracles
    T4 = "T4"  # Advanced: Error correction
    T5 = "T5"  # Research: Novel algorithms


@dataclass
class BenchmarkProgram:
    """A single benchmark program with ground truth specification."""
    
    id: str
    name: str
    tier: BenchmarkTier
    source: str
    language: str
    ground_truth_spec: Specification
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_quantum_program(self) -> QuantumProgram:
        """Convert to QuantumProgram instance."""
        if self.language.lower() == "silq":
            return QuantumProgram.from_silq(self.source, name=self.name)
        else:
            return QuantumProgram.from_openqasm(self.source, name=self.name)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.value,
            "source": self.source,
            "language": self.language,
            "ground_truth_spec": self.ground_truth_spec.to_dict(),
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BenchmarkProgram':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            tier=BenchmarkTier(data["tier"]),
            source=data["source"],
            language=data.get("language", "silq"),
            ground_truth_spec=Specification.from_dict(data["ground_truth_spec"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationResult:
    """Result of evaluating an LLM on a single program."""
    
    program_id: str
    synthesis_success: bool
    verification_success: bool
    specification_matches_ground_truth: bool
    time_seconds: float
    llm_calls: int
    error_message: str = ""
    synthesized_spec: Optional[Specification] = None


@dataclass
class BenchmarkResults:
    """Aggregated results from benchmark evaluation."""
    
    llm_model: str
    tier: str
    total_programs: int
    results: list[EvaluationResult]
    total_time_seconds: float
    
    @property
    def synthesis_rate(self) -> float:
        """Percentage of programs with successful synthesis."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.synthesis_success)
        return successes / len(self.results)
    
    @property
    def verification_rate(self) -> float:
        """Percentage of programs with successful verification."""
        if not self.results:
            return 0.0
        successes = sum(1 for r in self.results if r.verification_success)
        return successes / len(self.results)
    
    @property
    def accuracy(self) -> float:
        """Percentage matching ground truth."""
        if not self.results:
            return 0.0
        matches = sum(1 for r in self.results if r.specification_matches_ground_truth)
        return matches / len(self.results)
    
    @property
    def avg_time(self) -> float:
        """Average time per program."""
        if not self.results:
            return 0.0
        return sum(r.time_seconds for r in self.results) / len(self.results)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llm_model": self.llm_model,
            "tier": self.tier,
            "total_programs": self.total_programs,
            "synthesis_rate": self.synthesis_rate,
            "verification_rate": self.verification_rate,
            "accuracy": self.accuracy,
            "avg_time": self.avg_time,
            "total_time": self.total_time_seconds,
        }


class QVerifyBench:
    """
    Benchmark framework for quantum program verification.
    
    QVerifyBench provides 500+ quantum programs across five difficulty tiers
    with ground-truth specifications for evaluating LLM verification capabilities.
    
    Example:
        >>> bench = QVerifyBench(tier="T2")
        >>> results = bench.evaluate(llm="gpt-4o", timeout=60)
        >>> print(f"Synthesis Rate: {results.synthesis_rate:.2%}")
    """
    
    def __init__(
        self,
        tier: Optional[str] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize benchmark.
        
        Args:
            tier: Specific tier to load (T1-T5) or None for all
            data_dir: Directory containing benchmark data
        """
        self.tier = BenchmarkTier(tier) if tier else None
        self.data_dir = data_dir or Path(__file__).parent / "programs"
        self.programs: list[BenchmarkProgram] = []
        
        self._load_programs()
    
    def _load_programs(self) -> None:
        """Load benchmark programs."""
        # Load built-in programs
        self.programs = self._get_builtin_programs()
        
        # Filter by tier if specified
        if self.tier is not None:
            self.programs = [p for p in self.programs if p.tier == self.tier]
        
        # Try to load additional programs from data directory
        if self.data_dir.exists():
            for json_file in self.data_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            prog = BenchmarkProgram.from_dict(item)
                            if self.tier is None or prog.tier == self.tier:
                                self.programs.append(prog)
                    else:
                        prog = BenchmarkProgram.from_dict(data)
                        if self.tier is None or prog.tier == self.tier:
                            self.programs.append(prog)
                except Exception as e:
                    logger.debug(f"Could not load {json_file}: {e}")
    
    def _get_builtin_programs(self) -> list[BenchmarkProgram]:
        """Get built-in benchmark programs."""
        from qverify.core.specification import Precondition, Postcondition
        
        programs = []
        
        # T1: Basic programs
        programs.append(BenchmarkProgram(
            id="t1_hadamard",
            name="Hadamard Gate",
            tier=BenchmarkTier.T1,
            source="""def hadamard(q: quon) -> quon {
    return H(q);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("in_basis(q, |0⟩)"),
                postcondition=Postcondition.from_string("superposition(q)"),
                name="hadamard_spec",
            ),
            description="Apply Hadamard gate to create superposition",
            tags=["single-qubit", "superposition"],
        ))
        
        programs.append(BenchmarkProgram(
            id="t1_x_gate",
            name="Pauli-X Gate",
            tier=BenchmarkTier.T1,
            source="""def pauli_x(q: quon) -> quon {
    return X(q);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("in_basis(q, |0⟩)"),
                postcondition=Postcondition.from_string("in_basis(q, |1⟩)"),
                name="x_gate_spec",
            ),
            description="Apply Pauli-X (NOT) gate",
            tags=["single-qubit", "bit-flip"],
        ))
        
        programs.append(BenchmarkProgram(
            id="t1_measure",
            name="Measure Qubit",
            tier=BenchmarkTier.T1,
            source="""def measure_qubit(q: quon) -> bool {
    return measure(q);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("true"),
                postcondition=Postcondition.from_string("true"),
                name="measure_spec",
            ),
            description="Measure a qubit in computational basis",
            tags=["single-qubit", "measurement"],
        ))
        
        # T2: Multi-qubit programs
        programs.append(BenchmarkProgram(
            id="t2_bell_state",
            name="Bell State Preparation",
            tier=BenchmarkTier.T2,
            source="""def bell_state(q0: quon, q1: quon) -> (quon, quon) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("in_basis(q0, |0⟩) and in_basis(q1, |0⟩)"),
                postcondition=Postcondition.from_string("entangled(q0, q1)"),
                name="bell_state_spec",
            ),
            description="Prepare Bell state |Φ+⟩",
            tags=["multi-qubit", "entanglement", "bell-state"],
        ))
        
        programs.append(BenchmarkProgram(
            id="t2_swap",
            name="SWAP Gate",
            tier=BenchmarkTier.T2,
            source="""def swap_qubits(q0: quon, q1: quon) -> (quon, quon) {
    (q0, q1) = CNOT(q0, q1);
    (q1, q0) = CNOT(q1, q0);
    (q0, q1) = CNOT(q0, q1);
    return (q0, q1);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("true"),
                postcondition=Postcondition.from_string("true"),
                name="swap_spec",
            ),
            description="Swap two qubits using CNOTs",
            tags=["multi-qubit", "swap"],
        ))
        
        programs.append(BenchmarkProgram(
            id="t2_ghz",
            name="GHZ State",
            tier=BenchmarkTier.T2,
            source="""def ghz_state(q0: quon, q1: quon, q2: quon) -> (quon, quon, quon) {
    q0 = H(q0);
    (q0, q1) = CNOT(q0, q1);
    (q1, q2) = CNOT(q1, q2);
    return (q0, q1, q2);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("in_basis(q0, |0⟩) and in_basis(q1, |0⟩) and in_basis(q2, |0⟩)"),
                postcondition=Postcondition.from_string("entangled(q0, q1) and entangled(q1, q2)"),
                name="ghz_spec",
            ),
            description="Prepare 3-qubit GHZ state",
            tags=["multi-qubit", "entanglement", "ghz"],
        ))
        
        # T3: Programs with loops/oracles
        programs.append(BenchmarkProgram(
            id="t3_grover_iteration",
            name="Grover Iteration",
            tier=BenchmarkTier.T3,
            source="""def grover_iteration(qubits: []quon, oracle: []quon -> []quon) -> []quon {
    // Apply oracle
    qubits = oracle(qubits);
    // Apply diffusion
    for i in 0..len(qubits) {
        qubits[i] = H(qubits[i]);
    }
    // Phase flip
    for i in 0..len(qubits) {
        qubits[i] = X(qubits[i]);
    }
    // Multi-controlled Z
    qubits = MCZ(qubits);
    for i in 0..len(qubits) {
        qubits[i] = X(qubits[i]);
    }
    for i in 0..len(qubits) {
        qubits[i] = H(qubits[i]);
    }
    return qubits;
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("superposition(qubits)"),
                postcondition=Postcondition.from_string("prob(qubits, marked) >= 0.5"),
                name="grover_spec",
            ),
            description="Single Grover iteration with oracle and diffusion",
            tags=["algorithm", "grover", "oracle", "loop"],
        ))
        
        # T4: Error correction
        programs.append(BenchmarkProgram(
            id="t4_bit_flip_encode",
            name="Bit Flip Code Encoding",
            tier=BenchmarkTier.T4,
            source="""def bit_flip_encode(q: quon, a1: quon, a2: quon) -> (quon, quon, quon) {
    // Encode logical qubit into 3 physical qubits
    (q, a1) = CNOT(q, a1);
    (q, a2) = CNOT(q, a2);
    return (q, a1, a2);
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("in_basis(a1, |0⟩) and in_basis(a2, |0⟩)"),
                postcondition=Postcondition.from_string("entangled(q, a1) and entangled(q, a2)"),
                name="bit_flip_encode_spec",
            ),
            description="Encode qubit in 3-qubit bit flip code",
            tags=["error-correction", "encoding"],
        ))
        
        # T5: Research-level
        programs.append(BenchmarkProgram(
            id="t5_vqe_ansatz",
            name="VQE Ansatz Layer",
            tier=BenchmarkTier.T5,
            source="""def vqe_layer(qubits: []quon, params: []float) -> []quon {
    // Single-qubit rotations
    for i in 0..len(qubits) {
        qubits[i] = RY(params[i])(qubits[i]);
        qubits[i] = RZ(params[i + len(qubits)])(qubits[i]);
    }
    // Entangling layer
    for i in 0..len(qubits)-1 {
        (qubits[i], qubits[i+1]) = CNOT(qubits[i], qubits[i+1]);
    }
    return qubits;
}""",
            language="silq",
            ground_truth_spec=Specification(
                precondition=Precondition.from_string("true"),
                postcondition=Postcondition.from_string("true"),
                name="vqe_ansatz_spec",
            ),
            description="Single layer of hardware-efficient VQE ansatz",
            tags=["vqe", "variational", "ansatz"],
        ))
        
        return programs
    
    def __len__(self) -> int:
        return len(self.programs)
    
    def __iter__(self) -> Iterator[BenchmarkProgram]:
        return iter(self.programs)
    
    def get_program(self, program_id: str) -> Optional[BenchmarkProgram]:
        """Get a specific program by ID."""
        for prog in self.programs:
            if prog.id == program_id:
                return prog
        return None
    
    def evaluate(
        self,
        llm: str = "claude-3.5-sonnet",
        timeout: float = 60.0,
        qverify_instance: Any = None,
    ) -> BenchmarkResults:
        """
        Evaluate an LLM on the benchmark.
        
        Args:
            llm: LLM model identifier
            timeout: Timeout per program in seconds
            qverify_instance: Optional QVerify instance to use
            
        Returns:
            BenchmarkResults with aggregated metrics
        """
        from qverify import QVerify
        
        start_time = time.time()
        results = []
        
        # Create QVerify instance if not provided
        if qverify_instance is None:
            qverify_instance = QVerify(llm=llm)
        
        for prog in self.programs:
            result = self._evaluate_single(prog, qverify_instance, timeout)
            results.append(result)
            
            logger.info(
                f"[{prog.id}] Synthesis: {result.synthesis_success}, "
                f"Verification: {result.verification_success}, "
                f"Time: {result.time_seconds:.2f}s"
            )
        
        return BenchmarkResults(
            llm_model=llm,
            tier=self.tier.value if self.tier else "all",
            total_programs=len(self.programs),
            results=results,
            total_time_seconds=time.time() - start_time,
        )
    
    def _evaluate_single(
        self,
        prog: BenchmarkProgram,
        qverify: Any,
        timeout: float,
    ) -> EvaluationResult:
        """Evaluate a single program."""
        start_time = time.time()
        
        try:
            quantum_prog = prog.to_quantum_program()
            
            # Synthesize specification
            synth_result = qverify.synthesize_specification(quantum_prog)
            
            synthesis_success = synth_result.status == SynthesisStatus.SUCCESS
            
            if not synthesis_success:
                return EvaluationResult(
                    program_id=prog.id,
                    synthesis_success=False,
                    verification_success=False,
                    specification_matches_ground_truth=False,
                    time_seconds=time.time() - start_time,
                    llm_calls=getattr(synth_result, 'llm_calls', 0),
                    error_message=synth_result.message,
                )
            
            # Verify
            verify_result = qverify.verify(quantum_prog, synth_result.specification)
            verification_success = verify_result.status == VerificationStatus.VALID
            
            # Compare with ground truth
            matches_ground_truth = self._compare_specs(
                synth_result.specification,
                prog.ground_truth_spec,
            )
            
            return EvaluationResult(
                program_id=prog.id,
                synthesis_success=True,
                verification_success=verification_success,
                specification_matches_ground_truth=matches_ground_truth,
                time_seconds=time.time() - start_time,
                llm_calls=getattr(synth_result, 'llm_calls', 0),
                synthesized_spec=synth_result.specification,
            )
            
        except Exception as e:
            return EvaluationResult(
                program_id=prog.id,
                synthesis_success=False,
                verification_success=False,
                specification_matches_ground_truth=False,
                time_seconds=time.time() - start_time,
                llm_calls=0,
                error_message=str(e),
            )
    
    def _compare_specs(
        self,
        synthesized: Specification,
        ground_truth: Specification,
    ) -> bool:
        """Compare synthesized spec with ground truth."""
        # Simplified comparison - check if key predicates match
        synth_pre = synthesized.precondition.to_human_readable().lower()
        synth_post = synthesized.postcondition.to_human_readable().lower()
        gt_pre = ground_truth.precondition.to_human_readable().lower()
        gt_post = ground_truth.postcondition.to_human_readable().lower()
        
        # Extract key terms
        synth_terms = set(synth_pre.split() + synth_post.split())
        gt_terms = set(gt_pre.split() + gt_post.split())
        
        # Check overlap
        overlap = len(synth_terms & gt_terms) / max(len(gt_terms), 1)
        return overlap > 0.5
    
    def save_results(
        self,
        results: BenchmarkResults,
        path: Path,
    ) -> None:
        """Save benchmark results to JSON."""
        with open(path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get benchmark statistics."""
        tier_counts = {}
        for prog in self.programs:
            tier = prog.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        return {
            "total_programs": len(self.programs),
            "by_tier": tier_counts,
            "languages": list(set(p.language for p in self.programs)),
        }
