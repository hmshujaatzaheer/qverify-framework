"""
Formal specification types for quantum programs.

This module provides data structures for representing formal specifications
including preconditions, postconditions, and loop invariants for quantum programs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
import re


class Condition(ABC):
    """Abstract base class for specification conditions."""
    
    @abstractmethod
    def to_smt(self) -> str:
        """Convert condition to SMT-LIB format."""
        pass
    
    @abstractmethod
    def to_human_readable(self) -> str:
        """Convert condition to human-readable format."""
        pass
    
    @abstractmethod
    def substitute(self, mapping: dict[str, str]) -> 'Condition':
        """Substitute variables according to mapping."""
        pass


@dataclass
class AtomicCondition(Condition):
    """An atomic (non-compound) condition."""
    
    predicate: str
    arguments: list[str] = field(default_factory=list)
    
    def to_smt(self) -> str:
        """Convert to SMT-LIB format."""
        if not self.arguments:
            return self.predicate
        args_str = " ".join(self.arguments)
        return f"({self.predicate} {args_str})"
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        if not self.arguments:
            return self.predicate
        args_str = ", ".join(self.arguments)
        return f"{self.predicate}({args_str})"
    
    def substitute(self, mapping: dict[str, str]) -> 'AtomicCondition':
        """Substitute variables according to mapping."""
        new_args = [mapping.get(arg, arg) for arg in self.arguments]
        return AtomicCondition(predicate=self.predicate, arguments=new_args)


@dataclass
class CompoundCondition(Condition):
    """A compound condition (conjunction, disjunction, negation, etc.)."""
    
    operator: str  # "and", "or", "not", "implies", "iff"
    operands: list[Condition] = field(default_factory=list)
    
    def to_smt(self) -> str:
        """Convert to SMT-LIB format."""
        if self.operator == "not":
            return f"(not {self.operands[0].to_smt()})"
        operands_smt = " ".join(op.to_smt() for op in self.operands)
        return f"({self.operator} {operands_smt})"
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        if self.operator == "not":
            return f"¬({self.operands[0].to_human_readable()})"
        
        op_symbol = {
            "and": "∧",
            "or": "∨",
            "implies": "→",
            "iff": "↔"
        }.get(self.operator, self.operator)
        
        parts = [f"({op.to_human_readable()})" for op in self.operands]
        return f" {op_symbol} ".join(parts)
    
    def substitute(self, mapping: dict[str, str]) -> 'CompoundCondition':
        """Substitute variables according to mapping."""
        new_operands = [op.substitute(mapping) for op in self.operands]
        return CompoundCondition(operator=self.operator, operands=new_operands)


@dataclass
class QuantumCondition(Condition):
    """A condition specific to quantum states."""
    
    predicate_type: str  # "basis", "entangled", "amplitude", "probability", "superposition"
    qubits: list[str]
    value: Optional[Any] = None
    comparison: str = "="  # "=", "!=", "<", ">", "<=", ">="
    
    def to_smt(self) -> str:
        """Convert to SMT-LIB format."""
        qubits_str = " ".join(self.qubits)
        
        if self.predicate_type == "basis":
            return f"(in_basis {qubits_str} {self.value})"
        elif self.predicate_type == "entangled":
            return f"(entangled {qubits_str})"
        elif self.predicate_type == "amplitude":
            return f"({self.comparison} (amplitude {qubits_str}) {self.value})"
        elif self.predicate_type == "probability":
            return f"({self.comparison} (prob {qubits_str}) {self.value})"
        elif self.predicate_type == "superposition":
            return f"(superposition {qubits_str})"
        else:
            return f"({self.predicate_type} {qubits_str})"
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        qubits_str = ", ".join(self.qubits)
        
        if self.predicate_type == "basis":
            return f"{qubits_str} in basis {self.value}"
        elif self.predicate_type == "entangled":
            return f"entangled({qubits_str})"
        elif self.predicate_type == "amplitude":
            return f"amplitude({qubits_str}) {self.comparison} {self.value}"
        elif self.predicate_type == "probability":
            return f"P({qubits_str}) {self.comparison} {self.value}"
        elif self.predicate_type == "superposition":
            return f"superposition({qubits_str})"
        else:
            return f"{self.predicate_type}({qubits_str})"
    
    def substitute(self, mapping: dict[str, str]) -> 'QuantumCondition':
        """Substitute variables according to mapping."""
        new_qubits = [mapping.get(q, q) for q in self.qubits]
        return QuantumCondition(
            predicate_type=self.predicate_type,
            qubits=new_qubits,
            value=self.value,
            comparison=self.comparison
        )


@dataclass
class Precondition:
    """
    Precondition for a quantum program.
    
    Describes the required state of quantum registers before program execution.
    """
    
    condition: Condition
    description: str = ""
    
    @classmethod
    def from_string(cls, condition_str: str, description: str = "") -> 'Precondition':
        """Parse a precondition from a string representation."""
        condition = cls._parse_condition(condition_str)
        return cls(condition=condition, description=description)
    
    @classmethod
    def true(cls) -> 'Precondition':
        """Create a trivially true precondition."""
        return cls(
            condition=AtomicCondition(predicate="true"),
            description="No precondition required"
        )
    
    @classmethod
    def _parse_condition(cls, condition_str: str) -> Condition:
        """Parse a condition string into a Condition object."""
        condition_str = condition_str.strip()
        
        # Check for quantum predicates
        if condition_str.startswith("in_basis"):
            match = re.match(r'in_basis\s*\(([^,]+),\s*(.+)\)', condition_str)
            if match:
                return QuantumCondition(
                    predicate_type="basis",
                    qubits=[match.group(1).strip()],
                    value=match.group(2).strip()
                )
        
        if condition_str.startswith("entangled"):
            match = re.match(r'entangled\s*\(([^)]+)\)', condition_str)
            if match:
                qubits = [q.strip() for q in match.group(1).split(',')]
                return QuantumCondition(predicate_type="entangled", qubits=qubits)
        
        if "prob(" in condition_str or "probability(" in condition_str:
            match = re.match(r'prob(?:ability)?\s*\(([^)]+)\)\s*(>=|<=|>|<|=|!=)\s*(.+)', condition_str)
            if match:
                return QuantumCondition(
                    predicate_type="probability",
                    qubits=[match.group(1).strip()],
                    comparison=match.group(2),
                    value=float(match.group(3))
                )
        
        # Default to atomic condition
        return AtomicCondition(predicate=condition_str)
    
    def to_smt(self) -> str:
        """Convert precondition to SMT-LIB format."""
        return self.condition.to_smt()
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        return self.condition.to_human_readable()
    
    def __str__(self) -> str:
        return f"Pre: {self.to_human_readable()}"


@dataclass
class Postcondition:
    """
    Postcondition for a quantum program.
    
    Describes the guaranteed state of quantum registers after program execution.
    """
    
    condition: Condition
    description: str = ""
    
    @classmethod
    def from_string(cls, condition_str: str, description: str = "") -> 'Postcondition':
        """Parse a postcondition from a string representation."""
        condition = Precondition._parse_condition(condition_str)
        return cls(condition=condition, description=description)
    
    @classmethod
    def true(cls) -> 'Postcondition':
        """Create a trivially true postcondition."""
        return cls(
            condition=AtomicCondition(predicate="true"),
            description="No postcondition guaranteed"
        )
    
    def to_smt(self) -> str:
        """Convert postcondition to SMT-LIB format."""
        return self.condition.to_smt()
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        return self.condition.to_human_readable()
    
    def __str__(self) -> str:
        return f"Post: {self.to_human_readable()}"


@dataclass
class Invariant:
    """
    Loop invariant for iterative quantum algorithms.
    
    Describes a property that holds at each iteration of a loop.
    """
    
    condition: Condition
    loop_variable: str = ""
    description: str = ""
    
    @classmethod
    def from_string(
        cls, 
        condition_str: str, 
        loop_variable: str = "",
        description: str = ""
    ) -> 'Invariant':
        """Parse an invariant from a string representation."""
        condition = Precondition._parse_condition(condition_str)
        return cls(
            condition=condition, 
            loop_variable=loop_variable,
            description=description
        )
    
    def to_smt(self) -> str:
        """Convert invariant to SMT-LIB format."""
        return self.condition.to_smt()
    
    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        base = self.condition.to_human_readable()
        if self.loop_variable:
            return f"∀{self.loop_variable}: {base}"
        return base
    
    def __str__(self) -> str:
        return f"Inv: {self.to_human_readable()}"


@dataclass
class Specification:
    """
    Complete formal specification for a quantum program.
    
    A specification consists of:
    - Precondition: What must hold before execution
    - Postcondition: What is guaranteed after execution
    - Invariants: What holds at each loop iteration (if applicable)
    
    Example:
        >>> spec = Specification(
        ...     precondition=Precondition.from_string("in_basis(q0, |0⟩)"),
        ...     postcondition=Postcondition.from_string("superposition(q0)"),
        ...     name="hadamard_spec"
        ... )
        >>> print(spec)
        Specification(hadamard_spec)
          Pre: q0 in basis |0⟩
          Post: superposition(q0)
    """
    
    precondition: Precondition
    postcondition: Postcondition
    invariants: list[Invariant] = field(default_factory=list)
    name: str = "unnamed"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def trivial(cls, name: str = "trivial") -> 'Specification':
        """Create a trivial specification (true precondition and postcondition)."""
        return cls(
            precondition=Precondition.true(),
            postcondition=Postcondition.true(),
            name=name,
            description="Trivial specification"
        )
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Specification':
        """Create a specification from a dictionary."""
        pre = Precondition.from_string(data.get("precondition", "true"))
        post = Postcondition.from_string(data.get("postcondition", "true"))
        
        invariants = []
        for inv_str in data.get("invariants", []):
            invariants.append(Invariant.from_string(inv_str))
        
        return cls(
            precondition=pre,
            postcondition=post,
            invariants=invariants,
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "precondition": self.precondition.to_human_readable(),
            "postcondition": self.postcondition.to_human_readable(),
            "invariants": [inv.to_human_readable() for inv in self.invariants],
            "metadata": self.metadata
        }
    
    def to_smt(self) -> dict[str, str]:
        """Convert specification to SMT-LIB format."""
        return {
            "precondition": self.precondition.to_smt(),
            "postcondition": self.postcondition.to_smt(),
            "invariants": [inv.to_smt() for inv in self.invariants]
        }
    
    def is_complete(self) -> bool:
        """Check if specification is complete (non-trivial pre and post)."""
        pre_trivial = self.precondition.to_smt() == "true"
        post_trivial = self.postcondition.to_smt() == "true"
        return not (pre_trivial and post_trivial)
    
    def __str__(self) -> str:
        lines = [f"Specification({self.name})"]
        lines.append(f"  {self.precondition}")
        lines.append(f"  {self.postcondition}")
        for inv in self.invariants:
            lines.append(f"  {inv}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Specification(name={self.name}, complete={self.is_complete()})"
