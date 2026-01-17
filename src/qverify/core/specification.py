"""Specification representation for quantum programs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Precondition:
    """A precondition on quantum program input states."""

    formula: str
    variables: List[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_string(cls, formula: str) -> "Precondition":
        """Parse a precondition from string representation."""
        variables = list(set(re.findall(r'\b(q\d+|qubit\d*|[a-z]_?\d*)\b', formula)))
        variables = [v for v in variables if not v.startswith('in_') and v not in ['true', 'false']]
        return cls(formula=formula, variables=variables)

    def to_smt(self) -> str:
        """Convert to SMT-LIB2 format."""
        smt = self.formula
        smt = smt.replace("∧", " and ").replace("∨", " or ").replace("¬", "not ")
        smt = smt.replace("→", "=>").replace("↔", "=")
        smt = smt.replace("|0⟩", "zero").replace("|1⟩", "one")
        smt = smt.replace("|+⟩", "plus").replace("|-⟩", "minus")
        smt = re.sub(r',\s*', ' ', smt)
        if " and " in smt:
            parts = smt.split(" and ")
            smt = "(and " + " ".join(parts) + ")"
        return smt.strip()

    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        return self.description if self.description else self.formula

    def __str__(self) -> str:
        """Return string representation."""
        return self.formula


@dataclass
class Postcondition:
    """A postcondition on quantum program output states."""

    formula: str
    variables: List[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_string(cls, formula: str) -> "Postcondition":
        """Parse a postcondition from string representation."""
        variables = list(set(re.findall(r'\b(q\d+|qubit\d*|[a-z]_?\d*)\b', formula)))
        variables = [v for v in variables if not v.startswith('in_') and v not in ['true', 'false']]
        return cls(formula=formula, variables=variables)

    def to_smt(self) -> str:
        """Convert to SMT-LIB2 format."""
        smt = self.formula
        smt = smt.replace("∧", " and ").replace("∨", " or ").replace("¬", "not ")
        smt = smt.replace("→", "=>").replace("↔", "=")
        smt = smt.replace("|0⟩", "zero").replace("|1⟩", "one")
        smt = re.sub(r',\s*', ' ', smt)
        if " and " in smt:
            parts = smt.split(" and ")
            smt = "(and " + " ".join(parts) + ")"
        return smt.strip()

    def to_human_readable(self) -> str:
        """Convert to human-readable format."""
        return self.description if self.description else self.formula

    def __str__(self) -> str:
        """Return string representation."""
        return self.formula


@dataclass
class Invariant:
    """A loop invariant for iterative quantum algorithms."""

    formula: str
    location: str = "loop_entry"
    variables: List[str] = field(default_factory=list)

    @classmethod
    def from_string(cls, formula: str, location: str = "loop_entry") -> "Invariant":
        """Parse an invariant from string representation."""
        variables = list(set(re.findall(r'\b(q\d+|qubit\d*|[a-z]_?\d*)\b', formula)))
        variables = [v for v in variables if not v.startswith('prob') and v not in ['true', 'false']]
        return cls(formula=formula, location=location, variables=variables)

    def to_smt(self) -> str:
        """Convert to SMT-LIB2 format."""
        smt = self.formula
        smt = smt.replace(">=", " >= ").replace("<=", " <= ").replace("==", " = ")
        smt = re.sub(r',\s*', ' ', smt)
        return smt.strip()

    def __str__(self) -> str:
        """Return string representation."""
        return f"@{self.location}: {self.formula}"


@dataclass
class Specification:
    """A complete specification for a quantum program."""

    precondition: Precondition
    postcondition: Postcondition
    invariants: List[Invariant] = field(default_factory=list)
    name: str = "spec"
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Specification":
        """Create specification from dictionary."""
        pre = Precondition.from_string(data.get("precondition", "true"))
        post = Postcondition.from_string(data.get("postcondition", "true"))
        invariants = []
        for inv_data in data.get("invariants", []):
            if isinstance(inv_data, str):
                invariants.append(Invariant.from_string(inv_data))
            elif isinstance(inv_data, dict):
                invariants.append(Invariant.from_string(
                    inv_data.get("formula", "true"),
                    inv_data.get("location", "loop_entry")
                ))
        return cls(
            precondition=pre,
            postcondition=post,
            invariants=invariants,
            name=data.get("name", "spec"),
            description=data.get("description", data.get("reasoning", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "precondition": self.precondition.formula,
            "postcondition": self.postcondition.formula,
            "invariants": [inv.formula for inv in self.invariants],
            "description": self.description,
        }

    def __str__(self) -> str:
        """Return string representation."""
        inv_str = ", ".join(str(inv) for inv in self.invariants) if self.invariants else "none"
        return f"Specification({self.name}): Pre={self.precondition}, Post={self.postcondition}"
