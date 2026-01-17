"""QVerifyBench: Benchmark for quantum program verification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Specification
from qverify.core.types import BenchmarkResult

logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path(__file__).parent / "programs" / "programs.json"


@dataclass
class BenchmarkProgram:
    """A single program from the benchmark."""

    id: str
    tier: str
    name: str
    description: str
    source: str
    specification: Dict[str, Any]
    complexity: str
    gates: List[str]
    num_qubits: int

    def to_quantum_program(self) -> QuantumProgram:
        """Convert to QuantumProgram instance."""
        return QuantumProgram.from_silq(self.source)

    def to_specification(self) -> Specification:
        """Convert to Specification instance."""
        return Specification.from_dict(self.specification)


@dataclass
class QVerifyBench:
    """QVerifyBench benchmark suite with 500+ quantum programs."""

    tier: str = "all"
    benchmark_path: Optional[Path] = None
    _programs: List[BenchmarkProgram] = field(default_factory=list, init=False)
    _metadata: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Load benchmark data."""
        self._load_benchmark()

    def _load_benchmark(self):
        """Load benchmark programs from JSON."""
        path = self.benchmark_path or BENCHMARK_PATH

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            self._metadata = {
                "name": data.get("name", "QVerifyBench"),
                "version": data.get("version", "1.0.0"),
                "total_programs": data.get("total_programs", 0),
                "tiers": data.get("tiers", {}),
            }

            for prog_data in data.get("programs", []):
                program = BenchmarkProgram(
                    id=prog_data["id"],
                    tier=prog_data["tier"],
                    name=prog_data["name"],
                    description=prog_data["description"],
                    source=prog_data["source"],
                    specification=prog_data["specification"],
                    complexity=prog_data["complexity"],
                    gates=prog_data["gates"],
                    num_qubits=prog_data["num_qubits"],
                )

                if self.tier == "all" or program.tier == self.tier:
                    self._programs.append(program)

            logger.info(f"Loaded {len(self._programs)} programs from QVerifyBench")

        except FileNotFoundError:
            logger.warning(f"Benchmark file not found: {path}")
            self._programs = []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse benchmark: {e}")
            self._programs = []

    @property
    def programs(self) -> List[BenchmarkProgram]:
        """Get all loaded programs."""
        return self._programs

    @property
    def num_programs(self) -> int:
        """Get number of programs."""
        return len(self._programs)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get benchmark metadata."""
        return self._metadata

    def __iter__(self) -> Iterator[BenchmarkProgram]:
        """Iterate over programs."""
        return iter(self._programs)

    def __len__(self) -> int:
        """Get number of programs."""
        return len(self._programs)

    def __getitem__(self, idx: int) -> BenchmarkProgram:
        """Get program by index."""
        return self._programs[idx]

    def get_by_tier(self, tier: str) -> List[BenchmarkProgram]:
        """Get all programs in a tier."""
        return [p for p in self._programs if p.tier == tier]

    def evaluate(self, llm: str, timeout: float = 60.0) -> BenchmarkResult:
        """Evaluate an LLM on the benchmark."""
        return BenchmarkResult(
            synthesis_rate=0.0,
            verification_rate=0.0,
            avg_time=0.0,
            model=llm,
            total_programs=len(self._programs),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        stats = {"total_programs": len(self._programs), "by_tier": {}}
        for program in self._programs:
            tier = program.tier
            if tier not in stats["by_tier"]:
                stats["by_tier"][tier] = 0
            stats["by_tier"][tier] += 1
        return stats
