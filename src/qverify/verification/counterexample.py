"""
Counterexample Analysis for QVERIFY.

This module provides tools for analyzing counterexamples from failed
verification attempts, extracting concrete quantum states, and providing
diagnostic information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

from qverify.core.types import CounterExample, QuantumState, VerificationCondition


logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """Type of specification violation."""
    
    PRECONDITION_UNSATISFIED = auto()
    """Precondition not satisfied by input."""
    
    POSTCONDITION_VIOLATED = auto()
    """Postcondition violated by output."""
    
    INVARIANT_INITIATION = auto()
    """Loop invariant not established initially."""
    
    INVARIANT_CONSECUTION = auto()
    """Loop invariant not preserved by iteration."""
    
    INVARIANT_TERMINATION = auto()
    """Loop invariant doesn't imply postcondition."""
    
    FRAME_VIOLATION = auto()
    """Frame condition violated (unmodified qubit changed)."""
    
    UNKNOWN = auto()
    """Unknown violation type."""


@dataclass
class CounterexampleDiagnosis:
    """Diagnosis of a counterexample."""
    
    violation_type: ViolationType
    violated_clause: str
    input_description: str
    output_description: str
    suggested_fix: str
    confidence: float = 1.0


@dataclass
class CounterexampleAnalyzer:
    """
    Analyzer for verification counterexamples.
    
    Provides detailed analysis of why verification failed and suggests
    potential specification fixes.
    
    Example:
        >>> analyzer = CounterexampleAnalyzer()
        >>> diagnosis = analyzer.analyze(counterexample, vc)
        >>> print(diagnosis.suggested_fix)
    """
    
    def analyze(
        self,
        counterexample: CounterExample,
        vc: VerificationCondition,
    ) -> CounterexampleDiagnosis:
        """
        Analyze a counterexample and provide diagnosis.
        
        Args:
            counterexample: The counterexample to analyze
            vc: The verification condition that produced it
            
        Returns:
            Diagnosis with violation type and suggested fix
        """
        # Determine violation type from VC name
        violation_type = self._determine_violation_type(vc)
        
        # Describe input state
        input_desc = self._describe_state(counterexample.input_state)
        
        # Describe output state
        output_desc = self._describe_state(counterexample.output_state)
        
        # Generate suggested fix
        suggested_fix = self._suggest_fix(
            violation_type,
            counterexample,
            vc,
        )
        
        return CounterexampleDiagnosis(
            violation_type=violation_type,
            violated_clause=counterexample.violated_condition,
            input_description=input_desc,
            output_description=output_desc,
            suggested_fix=suggested_fix,
        )
    
    def _determine_violation_type(
        self,
        vc: VerificationCondition,
    ) -> ViolationType:
        """Determine violation type from VC."""
        name = vc.name.lower()
        
        if "pre" in name:
            return ViolationType.PRECONDITION_UNSATISFIED
        elif "post" in name or "hoare" in name:
            return ViolationType.POSTCONDITION_VIOLATED
        elif "initiation" in name:
            return ViolationType.INVARIANT_INITIATION
        elif "consecution" in name:
            return ViolationType.INVARIANT_CONSECUTION
        elif "termination" in name:
            return ViolationType.INVARIANT_TERMINATION
        elif "frame" in name:
            return ViolationType.FRAME_VIOLATION
        
        return ViolationType.UNKNOWN
    
    def _describe_state(self, state: QuantumState) -> str:
        """Generate human-readable description of quantum state."""
        if not hasattr(state, 'amplitudes') or state.amplitudes is None:
            return "Unknown state"
        
        n_qubits = state.num_qubits
        amps = state.amplitudes
        
        # Find significant amplitudes
        significant = []
        for i, amp in enumerate(amps):
            prob = np.abs(amp) ** 2
            if prob > 0.01:  # 1% threshold
                basis_str = format(i, f'0{n_qubits}b')
                significant.append(f"|{basis_str}⟩: {prob:.2%}")
        
        if not significant:
            return f"{n_qubits}-qubit state (no significant amplitudes)"
        
        return f"{n_qubits}-qubit state: " + ", ".join(significant[:5])
    
    def _suggest_fix(
        self,
        violation_type: ViolationType,
        counterexample: CounterExample,
        vc: VerificationCondition,
    ) -> str:
        """Suggest a fix based on violation type."""
        if violation_type == ViolationType.PRECONDITION_UNSATISFIED:
            return "Weaken the precondition to allow this input state"
        
        elif violation_type == ViolationType.POSTCONDITION_VIOLATED:
            return "Strengthen the precondition to exclude this input, or weaken the postcondition to allow this output"
        
        elif violation_type == ViolationType.INVARIANT_INITIATION:
            return "Weaken the loop invariant so it's established by the precondition"
        
        elif violation_type == ViolationType.INVARIANT_CONSECUTION:
            return "Strengthen the loop invariant to be preserved by the loop body"
        
        elif violation_type == ViolationType.INVARIANT_TERMINATION:
            return "Strengthen the loop invariant so it implies the postcondition when the loop exits"
        
        elif violation_type == ViolationType.FRAME_VIOLATION:
            return "Update the specification to account for modifications to this qubit"
        
        return "Review the specification for logical errors"
    
    def extract_concrete_input(
        self,
        model: dict[str, Any],
        num_qubits: int,
    ) -> QuantumState:
        """
        Extract a concrete input state from SMT model.
        
        Args:
            model: SMT model dictionary
            num_qubits: Number of qubits
            
        Returns:
            Concrete quantum state
        """
        # Try to extract amplitudes from model
        amplitudes = np.zeros(2 ** num_qubits, dtype=complex)
        
        # Look for amplitude values in model
        for key, value in model.items():
            if "amplitude" in key.lower():
                try:
                    # Parse index and value
                    idx = self._extract_index(key)
                    if idx is not None and idx < len(amplitudes):
                        amplitudes[idx] = complex(value)
                except (ValueError, TypeError):
                    pass
        
        # If no amplitudes found, create a default state
        if np.allclose(amplitudes, 0):
            # Default to |0...0⟩
            amplitudes[0] = 1.0
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 0:
            amplitudes /= norm
        
        return QuantumState(
            num_qubits=num_qubits,
            amplitudes=amplitudes,
        )
    
    def _extract_index(self, key: str) -> Optional[int]:
        """Extract index from model key."""
        import re
        match = re.search(r'\[(\d+)\]', key)
        if match:
            return int(match.group(1))
        return None


def create_counterexample_from_model(
    model: dict[str, Any],
    vc: VerificationCondition,
    num_qubits: int = 2,
) -> CounterExample:
    """
    Create a counterexample from an SMT model.
    
    Args:
        model: SMT model dictionary
        vc: The verification condition
        num_qubits: Number of qubits in the program
        
    Returns:
        CounterExample object
    """
    analyzer = CounterexampleAnalyzer()
    
    # Extract input state
    input_state = analyzer.extract_concrete_input(model, num_qubits)
    
    # For output state, we'd need to simulate - use placeholder
    output_state = QuantumState(
        num_qubits=num_qubits,
        amplitudes=np.zeros(2 ** num_qubits, dtype=complex),
    )
    output_state.amplitudes[0] = 1.0  # Placeholder
    
    return CounterExample(
        input_state=input_state,
        output_state=output_state,
        violated_condition=vc.name,
        trace=[vc.formula],
        additional_info={"model": model},
    )


def format_counterexample(cex: CounterExample) -> str:
    """
    Format a counterexample for display.
    
    Args:
        cex: The counterexample
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 50,
        "COUNTEREXAMPLE",
        "=" * 50,
        "",
        f"Violated Condition: {cex.violated_condition}",
        "",
        "Input State:",
    ]
    
    if hasattr(cex.input_state, 'amplitudes'):
        for i, amp in enumerate(cex.input_state.amplitudes):
            prob = np.abs(amp) ** 2
            if prob > 0.01:
                basis = format(i, f'0{cex.input_state.num_qubits}b')
                lines.append(f"  |{basis}⟩: amplitude={amp:.4f}, prob={prob:.2%}")
    
    lines.extend([
        "",
        "Execution Trace:",
    ])
    
    for step in cex.trace[:5]:  # Limit trace length
        lines.append(f"  {step[:80]}...")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
