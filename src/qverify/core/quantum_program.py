"""
Quantum Program representation for QVERIFY.

This module provides the QuantumProgram class which represents quantum programs
in various formats (Silq, OpenQASM, Qiskit) with parsing and analysis capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator, Optional, Union
import re


class ProgramLanguage(Enum):
    """Supported quantum programming languages."""
    
    SILQ = auto()
    OPENQASM = auto()
    QISKIT = auto()
    CIRQ = auto()


@dataclass
class Qubit:
    """Representation of a qubit in a quantum program."""
    
    name: str
    index: Optional[int] = None
    is_ancilla: bool = False
    
    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.name}[{self.index}]"
        return self.name
    
    def __hash__(self) -> int:
        return hash((self.name, self.index))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Qubit):
            return False
        return self.name == other.name and self.index == other.index


@dataclass
class Gate:
    """Representation of a quantum gate operation."""
    
    name: str
    qubits: list[Qubit]
    parameters: list[float] = field(default_factory=list)
    controls: list[Qubit] = field(default_factory=list)
    line_number: int = 0
    
    def __str__(self) -> str:
        qubit_str = ", ".join(str(q) for q in self.qubits)
        if self.parameters:
            param_str = ", ".join(f"{p:.4f}" for p in self.parameters)
            return f"{self.name}({param_str}) {qubit_str}"
        return f"{self.name} {qubit_str}"
    
    @property
    def is_measurement(self) -> bool:
        """Check if this gate is a measurement."""
        return self.name.upper() in {"M", "MEASURE", "MEASUREMENT"}
    
    @property
    def is_controlled(self) -> bool:
        """Check if this is a controlled gate."""
        return len(self.controls) > 0
    
    @property
    def num_qubits(self) -> int:
        """Get total number of qubits involved."""
        return len(self.qubits) + len(self.controls)


@dataclass
class Loop:
    """Representation of a loop in a quantum program."""
    
    variable: str
    start: int
    end: int
    body: list[Union[Gate, 'Loop']]
    line_number: int = 0
    
    def __str__(self) -> str:
        return f"for {self.variable} in {self.start}..{self.end}"


@dataclass
class Conditional:
    """Representation of a conditional in a quantum program."""
    
    condition: str
    then_branch: list[Union[Gate, Loop, 'Conditional']]
    else_branch: list[Union[Gate, Loop, 'Conditional']] = field(default_factory=list)
    line_number: int = 0


@dataclass
class Function:
    """Representation of a function/subroutine in a quantum program."""
    
    name: str
    parameters: list[tuple[str, str]]  # (name, type)
    return_type: str
    body: list[Union[Gate, Loop, Conditional]]
    line_number: int = 0
    
    def __str__(self) -> str:
        params = ", ".join(f"{name}: {typ}" for name, typ in self.parameters)
        return f"def {self.name}({params}) -> {self.return_type}"


@dataclass
class QuantumProgram:
    """
    Main representation of a quantum program.
    
    This class provides a unified interface for working with quantum programs
    from various languages (Silq, OpenQASM, Qiskit, etc.).
    
    Example:
        >>> program = QuantumProgram.from_silq('''
        ...     def bell_state(q0: qubit, q1: qubit) {
        ...         q0 = H(q0);
        ...         (q0, q1) = CNOT(q0, q1);
        ...         return (q0, q1);
        ...     }
        ... ''')
        >>> print(program.num_qubits)
        2
    """
    
    source: str
    language: ProgramLanguage
    name: str = "unnamed"
    
    # Parsed components
    qubits: list[Qubit] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Parse the program after initialization."""
        if not self.gates and not self.functions:
            self._parse()
    
    def _parse(self) -> None:
        """Parse the source code based on language."""
        if self.language == ProgramLanguage.SILQ:
            self._parse_silq()
        elif self.language == ProgramLanguage.OPENQASM:
            self._parse_openqasm()
        else:
            # Basic parsing for other languages
            self._parse_generic()
    
    def _parse_silq(self) -> None:
        """Parse Silq source code."""
        # Extract function definitions
        func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)\s*(?:->?\s*(\w+))?\s*\{'
        matches = re.finditer(func_pattern, self.source)
        
        for match in matches:
            name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3) or "void"
            
            # Parse parameters
            params = []
            if params_str.strip():
                for param in params_str.split(','):
                    parts = param.strip().split(':')
                    if len(parts) == 2:
                        params.append((parts[0].strip(), parts[1].strip()))
                    else:
                        params.append((parts[0].strip(), "qubit"))
            
            self.functions.append(Function(
                name=name,
                parameters=params,
                return_type=return_type,
                body=[],
                line_number=self.source[:match.start()].count('\n') + 1
            ))
        
        # Extract qubit declarations
        qubit_pattern = r'(\w+)\s*:\s*(?:!?\[?\]?)?qubit'
        for match in re.finditer(qubit_pattern, self.source):
            self.qubits.append(Qubit(name=match.group(1)))
        
        # Extract gate applications
        gate_pattern = r'(\w+)\s*=\s*(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(gate_pattern, self.source):
            result = match.group(1)
            gate_name = match.group(2)
            args = match.group(3)
            
            if gate_name.upper() in {'H', 'X', 'Y', 'Z', 'S', 'T', 'CNOT', 'CZ', 'SWAP', 'HADAMARD'}:
                qubits = [Qubit(name=arg.strip()) for arg in args.split(',') if arg.strip()]
                self.gates.append(Gate(
                    name=gate_name.upper(),
                    qubits=qubits,
                    line_number=self.source[:match.start()].count('\n') + 1
                ))
    
    def _parse_openqasm(self) -> None:
        """Parse OpenQASM source code."""
        # Extract qubit declarations
        qreg_pattern = r'qreg\s+(\w+)\s*\[(\d+)\]'
        for match in re.finditer(qreg_pattern, self.source):
            name = match.group(1)
            size = int(match.group(2))
            for i in range(size):
                self.qubits.append(Qubit(name=name, index=i))
        
        # Extract gate applications
        gate_pattern = r'(\w+)\s+([^;]+);'
        for match in re.finditer(gate_pattern, self.source):
            gate_name = match.group(1)
            args = match.group(2)
            
            if gate_name.lower() not in {'qreg', 'creg', 'include', 'openqasm'}:
                # Parse qubit arguments
                qubit_refs = re.findall(r'(\w+)\[(\d+)\]', args)
                qubits = [Qubit(name=name, index=int(idx)) for name, idx in qubit_refs]
                
                if qubits:
                    self.gates.append(Gate(
                        name=gate_name.upper(),
                        qubits=qubits,
                        line_number=self.source[:match.start()].count('\n') + 1
                    ))
    
    def _parse_generic(self) -> None:
        """Generic parsing for unsupported languages."""
        # Count qubits mentioned in source
        qubit_mentions = re.findall(r'q(\d+)|qubit(\d+)|qubits?\[(\d+)\]', self.source)
        max_idx = 0
        for match in qubit_mentions:
            for group in match:
                if group:
                    max_idx = max(max_idx, int(group) + 1)
        
        for i in range(max(max_idx, 1)):
            self.qubits.append(Qubit(name="q", index=i))
    
    @classmethod
    def from_silq(cls, source: str, name: str = "unnamed") -> QuantumProgram:
        """Create a QuantumProgram from Silq source code."""
        return cls(source=source, language=ProgramLanguage.SILQ, name=name)
    
    @classmethod
    def from_openqasm(cls, source: str, name: str = "unnamed") -> QuantumProgram:
        """Create a QuantumProgram from OpenQASM source code."""
        return cls(source=source, language=ProgramLanguage.OPENQASM, name=name)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> QuantumProgram:
        """Load a QuantumProgram from a file."""
        path = Path(path)
        source = path.read_text()
        
        # Determine language from extension
        ext = path.suffix.lower()
        if ext in {'.silq', '.sq'}:
            language = ProgramLanguage.SILQ
        elif ext in {'.qasm', '.openqasm'}:
            language = ProgramLanguage.OPENQASM
        else:
            language = ProgramLanguage.SILQ  # Default
        
        return cls(source=source, language=language, name=path.stem)
    
    @property
    def num_qubits(self) -> int:
        """Get the number of qubits in the program."""
        return len(self.qubits)
    
    @property
    def num_gates(self) -> int:
        """Get the number of gates in the program."""
        return len(self.gates)
    
    @property
    def depth(self) -> int:
        """Estimate the circuit depth."""
        # Simple depth estimation
        if not self.gates:
            return 0
        return len(self.gates)  # Simplified; proper implementation would track parallelism
    
    @property
    def has_loops(self) -> bool:
        """Check if the program contains loops."""
        return len(self.loops) > 0 or 'for' in self.source or 'while' in self.source
    
    @property
    def has_measurements(self) -> bool:
        """Check if the program contains measurements."""
        return any(g.is_measurement for g in self.gates) or 'measure' in self.source.lower()
    
    def get_gate_sequence(self) -> Iterator[Gate]:
        """Iterate over all gates in program order."""
        yield from self.gates
    
    def get_qubits_used_by_gate(self, gate: Gate) -> set[Qubit]:
        """Get all qubits used by a specific gate."""
        return set(gate.qubits) | set(gate.controls)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert program to dictionary representation."""
        return {
            "name": self.name,
            "language": self.language.name,
            "num_qubits": self.num_qubits,
            "num_gates": self.num_gates,
            "source": self.source,
            "qubits": [str(q) for q in self.qubits],
            "gates": [str(g) for g in self.gates],
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        return f"QuantumProgram(name={self.name}, qubits={self.num_qubits}, gates={self.num_gates})"
    
    def __repr__(self) -> str:
        return self.__str__()
