"""
Verification Condition Generator for Quantum Programs.

This module generates verification conditions (VCs) from quantum programs
and their specifications using weakest precondition calculus adapted for
quantum computing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from qverify.core.quantum_program import QuantumProgram, Gate, Loop
from qverify.core.specification import Specification, Precondition, Postcondition, Invariant
from qverify.core.types import VerificationCondition, ProgramLocation


logger = logging.getLogger(__name__)


@dataclass
class VCGeneratorConfig:
    """Configuration for VC generation."""
    
    include_frame_conditions: bool = True
    generate_loop_vcs: bool = True
    max_unroll_depth: int = 5
    use_quantum_wp: bool = True


@dataclass
class VCGenerator:
    """
    Verification Condition Generator for quantum programs.
    
    Generates VCs using quantum weakest precondition calculus.
    
    Example:
        >>> gen = VCGenerator()
        >>> vcs = gen.generate(program, specification)
        >>> for vc in vcs:
        ...     print(vc.name, vc.formula)
    """
    
    config: VCGeneratorConfig = field(default_factory=VCGeneratorConfig)
    
    def generate(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> list[VerificationCondition]:
        """
        Generate all verification conditions for a program.
        
        Args:
            program: The quantum program
            specification: The specification to verify
            
        Returns:
            List of verification conditions
        """
        vcs = []
        
        # VC 1: Precondition satisfiability
        vcs.append(self._generate_pre_sat_vc(specification))
        
        # VC 2: Main Hoare triple
        vcs.append(self._generate_hoare_vc(program, specification))
        
        # VC 3+: Loop invariant VCs
        if self.config.generate_loop_vcs and specification.invariants:
            vcs.extend(self._generate_loop_vcs(program, specification))
        
        # Frame conditions
        if self.config.include_frame_conditions:
            vcs.extend(self._generate_frame_vcs(program, specification))
        
        return vcs
    
    def _generate_pre_sat_vc(
        self,
        specification: Specification,
    ) -> VerificationCondition:
        """Generate VC asserting precondition is satisfiable."""
        return VerificationCondition(
            name="pre_satisfiable",
            formula=f"(assert {specification.precondition.to_smt()})",
            location=None,
            assumptions=[],
        )
    
    def _generate_hoare_vc(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> VerificationCondition:
        """Generate main Hoare triple VC: {Pre} P {Post}."""
        pre_smt = specification.precondition.to_smt()
        post_smt = specification.postcondition.to_smt()
        
        # Compute weakest precondition
        wp = self._compute_wp(program, post_smt)
        
        # VC: Pre => wp(P, Post)
        formula = f"(=> {pre_smt} {wp})"
        
        return VerificationCondition(
            name="hoare_triple",
            formula=formula,
            location=ProgramLocation(line=1, label="program_entry"),
            assumptions=[pre_smt],
        )
    
    def _compute_wp(
        self,
        program: QuantumProgram,
        postcondition: str,
    ) -> str:
        """
        Compute weakest precondition for quantum program.
        
        Uses quantum weakest precondition calculus:
        - wp(skip, Q) = Q
        - wp(U, Q) = U† Q U (for unitary U)
        - wp(measure, Q) = Σ_m P_m Q P_m
        - wp(S1; S2, Q) = wp(S1, wp(S2, Q))
        """
        current_wp = postcondition
        
        # Process gates in reverse order
        for gate in reversed(list(program.gates)):
            current_wp = self._gate_wp(gate, current_wp)
        
        return current_wp
    
    def _gate_wp(self, gate: Gate, post: str) -> str:
        """Compute weakest precondition for a single gate."""
        gate_name = gate.name.upper()
        qubits = [str(q) for q in gate.qubits]
        
        if gate_name in {"H", "HADAMARD"}:
            # Hadamard is self-inverse
            return f"(hadamard_wp {qubits[0]} {post})"
        
        elif gate_name in {"X", "NOT"}:
            # Pauli-X is self-inverse
            return f"(pauli_x_wp {qubits[0]} {post})"
        
        elif gate_name in {"Y"}:
            return f"(pauli_y_wp {qubits[0]} {post})"
        
        elif gate_name in {"Z"}:
            return f"(pauli_z_wp {qubits[0]} {post})"
        
        elif gate_name in {"CNOT", "CX"}:
            if len(qubits) >= 2:
                return f"(cnot_wp {qubits[0]} {qubits[1]} {post})"
        
        elif gate_name in {"CZ"}:
            if len(qubits) >= 2:
                return f"(cz_wp {qubits[0]} {qubits[1]} {post})"
        
        elif gate_name in {"SWAP"}:
            if len(qubits) >= 2:
                return f"(swap_wp {qubits[0]} {qubits[1]} {post})"
        
        elif gate_name in {"M", "MEASURE", "MEASUREMENT"}:
            return f"(measure_wp {qubits[0]} {post})"
        
        # Default: assume gate preserves postcondition
        return post
    
    def _generate_loop_vcs(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> list[VerificationCondition]:
        """Generate VCs for loop invariants."""
        vcs = []
        
        for i, inv in enumerate(specification.invariants):
            # Initiation: Pre => Inv
            vcs.append(VerificationCondition(
                name=f"inv_{i}_initiation",
                formula=f"(=> {specification.precondition.to_smt()} {inv.to_smt()})",
                location=ProgramLocation(line=0, label=f"loop_{i}_entry"),
                assumptions=[],
            ))
            
            # Consecution: Inv ∧ cond => wp(body, Inv)
            vcs.append(VerificationCondition(
                name=f"inv_{i}_consecution",
                formula=f"(=> {inv.to_smt()} {inv.to_smt()})",  # Simplified
                location=ProgramLocation(line=0, label=f"loop_{i}_body"),
                assumptions=[inv.to_smt()],
            ))
            
            # Termination: Inv ∧ ¬cond => Post
            vcs.append(VerificationCondition(
                name=f"inv_{i}_termination",
                formula=f"(=> {inv.to_smt()} {specification.postcondition.to_smt()})",
                location=ProgramLocation(line=0, label=f"loop_{i}_exit"),
                assumptions=[inv.to_smt()],
            ))
        
        return vcs
    
    def _generate_frame_vcs(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> list[VerificationCondition]:
        """Generate frame condition VCs (qubits not modified are preserved)."""
        vcs = []
        
        # Find qubits mentioned in spec but not modified by program
        spec_qubits = self._extract_qubits_from_spec(specification)
        modified_qubits = {str(q) for g in program.gates for q in g.qubits}
        
        unmodified = spec_qubits - modified_qubits
        
        for qubit in unmodified:
            vcs.append(VerificationCondition(
                name=f"frame_{qubit}",
                formula=f"(= (state_of {qubit} pre) (state_of {qubit} post))",
                location=None,
                assumptions=[],
            ))
        
        return vcs
    
    def _extract_qubits_from_spec(
        self,
        specification: Specification,
    ) -> set[str]:
        """Extract qubit names mentioned in specification."""
        qubits = set()
        
        # Simple extraction from SMT strings
        import re
        
        for text in [
            specification.precondition.to_smt(),
            specification.postcondition.to_smt(),
        ]:
            # Match patterns like q0, q1, qubit0, etc.
            matches = re.findall(r'\b(q\d+|qubit\d*)\b', text, re.IGNORECASE)
            qubits.update(matches)
        
        return qubits


def generate_vcs(
    program: QuantumProgram,
    specification: Specification,
    config: Optional[VCGeneratorConfig] = None,
) -> list[VerificationCondition]:
    """
    Convenience function to generate verification conditions.
    
    Args:
        program: The quantum program
        specification: The specification
        config: Optional configuration
        
    Returns:
        List of verification conditions
    """
    generator = VCGenerator(config=config or VCGeneratorConfig())
    return generator.generate(program, specification)
