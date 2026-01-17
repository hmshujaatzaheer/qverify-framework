"""Quantum program representation and parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qverify.core.types import Gate


@dataclass
class QuantumProgram:
    """Representation of a quantum program."""

    source: str
    language: str = "silq"
    name: str = "program"
    qubits: List[str] = field(default_factory=list)
    gates: List[Gate] = field(default_factory=list)
    num_qubits: int = 0
    has_loops: bool = False
    has_measurements: bool = False
    parameters: List[str] = field(default_factory=list)
    return_type: str = "qubit"

    @classmethod
    def from_silq(cls, source: str) -> "QuantumProgram":
        """Parse a Silq program."""
        program = cls(source=source, language="silq")
        program._parse_silq()
        return program

    @classmethod
    def from_qasm(cls, source: str) -> "QuantumProgram":
        """Parse an OpenQASM program."""
        program = cls(source=source, language="qasm")
        program._parse_qasm()
        return program

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuantumProgram":
        """Create from dictionary representation."""
        gates = [
            Gate(
                name=g["gate"],
                qubits=g.get("qubits", []),
                params=g.get("params", []),
                line_number=g.get("line", 0)
            )
            for g in data.get("gates", [])
        ]
        return cls(
            source=data.get("source", ""),
            language=data.get("language", "silq"),
            name=data.get("name", "program"),
            qubits=data.get("qubits", []),
            gates=gates,
            num_qubits=data.get("num_qubits", len(data.get("qubits", []))),
            has_loops=data.get("has_loops", False),
            has_measurements=data.get("has_measurements", False),
        )

    def _parse_silq(self) -> None:
        """Parse Silq source code."""
        func_match = re.search(r'def\s+(\w+)', self.source)
        if func_match:
            self.name = func_match.group(1)

        param_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', self.source)
        if param_match:
            params = param_match.group(1)
            for param in params.split(','):
                param = param.strip()
                if ':' in param:
                    name, type_info = param.split(':', 1)
                    name = name.strip()
                    type_info = type_info.strip()
                    if 'qubit' in type_info:
                        self.qubits.append(name)
                        if '[]' in type_info:
                            self.num_qubits += 4
                        else:
                            self.num_qubits += 1

        if not self.qubits:
            qubit_decls = re.findall(r'(\w+)\s*:\s*qubit', self.source)
            self.qubits = list(set(qubit_decls))
            self.num_qubits = len(self.qubits)

        self._extract_gates_silq()
        self.has_loops = bool(re.search(r'\b(for|while|repeat)\b', self.source))
        self.has_measurements = bool(re.search(r'\bmeasure\b', self.source, re.IGNORECASE))

    def _extract_gates_silq(self) -> None:
        """Extract gates from Silq source."""
        single_gates = ['H', 'X', 'Y', 'Z', 'S', 'T', 'hadamard']
        two_gates = ['CNOT', 'CX', 'CZ', 'SWAP']
        param_gates = ['RX', 'RY', 'RZ', 'U', 'phase']

        line_num = 0
        for line in self.source.split('\n'):
            line_num += 1
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            for gate in single_gates:
                pattern = rf'\b{gate}\s*\(\s*(\w+)'
                matches = re.findall(pattern, line, re.IGNORECASE)
                for qubit in matches:
                    self.gates.append(Gate(
                        name=gate.upper(),
                        qubits=[qubit],
                        line_number=line_num
                    ))

            for gate in two_gates:
                pattern = rf'\b{gate}\s*\(\s*(\w+)\s*,\s*(\w+)'
                matches = re.findall(pattern, line, re.IGNORECASE)
                for q1, q2 in matches:
                    self.gates.append(Gate(
                        name=gate.upper(),
                        qubits=[q1, q2],
                        line_number=line_num
                    ))

            for gate in param_gates:
                pattern = rf'\b{gate}\s*\(\s*([\d.]+)\s*,\s*(\w+)'
                matches = re.findall(pattern, line, re.IGNORECASE)
                for param, qubit in matches:
                    self.gates.append(Gate(
                        name=gate.upper(),
                        qubits=[qubit],
                        params=[float(param)],
                        line_number=line_num
                    ))

    def _parse_qasm(self) -> None:
        """Parse OpenQASM source code."""
        qreg_matches = re.findall(r'qreg\s+(\w+)\s*\[\s*(\d+)\s*\]', self.source)
        for name, size in qreg_matches:
            for i in range(int(size)):
                self.qubits.append(f"{name}[{i}]")
        self.num_qubits = len(self.qubits)
        self._extract_gates_qasm()
        self.has_loops = bool(re.search(r'\b(for|while)\b', self.source))
        self.has_measurements = bool(re.search(r'\bmeasure\b', self.source))

    def _extract_gates_qasm(self) -> None:
        """Extract gates from OpenQASM source."""
        line_num = 0
        for line in self.source.split('\n'):
            line_num += 1
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('OPENQASM'):
                continue
            if line.startswith('include') or line.startswith('qreg') or line.startswith('creg'):
                continue

            gate_match = re.match(r'(\w+)\s+(.+);', line)
            if gate_match:
                gate_name = gate_match.group(1).upper()
                qubits_str = gate_match.group(2)
                qubits = [q.strip() for q in qubits_str.split(',')]
                self.gates.append(Gate(
                    name=gate_name,
                    qubits=qubits,
                    line_number=line_num
                ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "language": self.language,
            "name": self.name,
            "qubits": self.qubits,
            "gates": [
                {"gate": g.name, "qubits": g.qubits, "params": g.params, "line": g.line_number}
                for g in self.gates
            ],
            "num_qubits": self.num_qubits,
            "has_loops": self.has_loops,
            "has_measurements": self.has_measurements,
        }

    def get_gate_sequence(self) -> List[str]:
        """Get list of gate names in order."""
        return [g.name for g in self.gates]

    def get_entangling_gates(self) -> List[Gate]:
        """Get all two-qubit (entangling) gates."""
        return [g for g in self.gates if len(g.qubits) >= 2]

    def __str__(self) -> str:
        """Return string representation."""
        return f"QuantumProgram(name={self.name}, qubits={self.num_qubits}, gates={len(self.gates)})"

    def __repr__(self) -> str:
        """Return debug representation."""
        return self.__str__()
