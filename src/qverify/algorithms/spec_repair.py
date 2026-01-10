"""
RepairSpecification: Counterexample-Guided Specification Refinement.

This module implements Algorithm 4 from the QVERIFY paper - repairing failed
specifications using counterexamples from the verifier.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import (
    Specification,
    Precondition,
    Postcondition,
    AtomicCondition,
    CompoundCondition,
)
from qverify.core.types import CounterExample


logger = logging.getLogger(__name__)


class FailureDiagnosis(Enum):
    """Types of specification failures."""
    
    WEAK_PRECONDITION = auto()
    """Precondition is too weak - allows invalid inputs."""
    
    STRONG_POSTCONDITION = auto()
    """Postcondition is too strong - excludes valid outputs."""
    
    MISSING_INVARIANT = auto()
    """Loop invariant is missing or incomplete."""
    
    LOGIC_ERROR = auto()
    """Logical error in specification structure."""
    
    UNKNOWN = auto()
    """Cannot determine failure cause."""


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""
    
    def generate(self, prompt: str, n: int = 1) -> list[str]:
        """Generate completions."""
        ...


REPAIR_PROMPT = """You are an expert in formal specification repair. A specification has failed verification.

## Program
```
{program_source}
```

## Failed Specification
Precondition: {precondition}
Postcondition: {postcondition}

## Counterexample
Input: {cex_input}
Output: {cex_output}
Violated: {violated_condition}

## Diagnosis
The failure appears to be: {diagnosis}

## Task
Repair the specification to handle this counterexample:
- If WEAK_PRECONDITION: Strengthen precondition to exclude bad inputs
- If STRONG_POSTCONDITION: Weaken postcondition to allow valid outputs
- If LOGIC_ERROR: Fix the logical structure

Output repaired specification as JSON:
```json
{{
    "precondition": "repaired precondition",
    "postcondition": "repaired postcondition",
    "repair_type": "strengthen_pre|weaken_post|restructure",
    "reasoning": "why this repair works"
}}
```"""


@dataclass
class RepairSpecification:
    """
    Repair specifications using counterexamples.
    
    This class implements counterexample-guided specification repair (CEGIS-like)
    to iteratively fix specifications that fail verification.
    
    Example:
        >>> repairer = RepairSpecification(llm=my_llm)
        >>> fixed_spec = repairer.repair(program, failed_spec, counterexample)
    """
    
    llm: Optional[LLMInterface] = None
    max_repair_attempts: int = 3
    
    def repair(
        self,
        program: QuantumProgram,
        specification: Specification,
        counterexample: CounterExample,
    ) -> Optional[Specification]:
        """
        Repair a specification given a counterexample.
        
        Args:
            program: The quantum program
            specification: The failed specification
            counterexample: Counterexample from verifier
            
        Returns:
            Repaired specification or None if repair fails
        """
        # Step 1: Diagnose the failure
        diagnosis = self._diagnose_failure(
            program, specification, counterexample
        )
        
        # Step 2: Apply repair strategy based on diagnosis
        if diagnosis == FailureDiagnosis.WEAK_PRECONDITION:
            return self._strengthen_precondition(
                program, specification, counterexample
            )
        elif diagnosis == FailureDiagnosis.STRONG_POSTCONDITION:
            return self._weaken_postcondition(
                program, specification, counterexample
            )
        else:
            return self._llm_repair(
                program, specification, counterexample, diagnosis
            )
    
    def _diagnose_failure(
        self,
        program: QuantumProgram,
        specification: Specification,
        counterexample: CounterExample,
    ) -> FailureDiagnosis:
        """Diagnose the cause of specification failure."""
        violated = counterexample.violated_condition.lower()
        
        # Check if precondition was satisfied but output was wrong
        if "postcondition" in violated or "post" in violated:
            # Could be strong postcondition or logic error
            # Check if the counterexample output seems reasonable
            return FailureDiagnosis.STRONG_POSTCONDITION
        
        if "precondition" in violated or "pre" in violated:
            return FailureDiagnosis.WEAK_PRECONDITION
        
        if "invariant" in violated or "loop" in violated:
            return FailureDiagnosis.MISSING_INVARIANT
        
        # Default: try to infer from counterexample structure
        return self._infer_diagnosis(counterexample)
    
    def _infer_diagnosis(self, counterexample: CounterExample) -> FailureDiagnosis:
        """Infer diagnosis from counterexample structure."""
        # If input looks unusual, probably weak precondition
        input_state = counterexample.input_state
        if hasattr(input_state, 'amplitudes'):
            import numpy as np
            probs = np.abs(input_state.amplitudes) ** 2
            
            # Check if input is a standard basis state or simple superposition
            max_prob = np.max(probs)
            if max_prob < 0.5:
                # Complex superposition input - might be precondition issue
                return FailureDiagnosis.WEAK_PRECONDITION
        
        # Default to postcondition issue
        return FailureDiagnosis.STRONG_POSTCONDITION
    
    def _strengthen_precondition(
        self,
        program: QuantumProgram,
        specification: Specification,
        counterexample: CounterExample,
    ) -> Optional[Specification]:
        """Strengthen precondition to exclude counterexample input."""
        # Create negation of counterexample input pattern
        exclusion = self._create_input_exclusion(counterexample)
        
        if exclusion is None:
            return None
        
        # Conjoin with existing precondition
        old_pre = specification.precondition.condition
        new_pre_condition = CompoundCondition(
            operator="and",
            operands=[old_pre, exclusion]
        )
        
        new_pre = Precondition(
            condition=new_pre_condition,
            description=f"Strengthened: {specification.precondition.description}"
        )
        
        return Specification(
            precondition=new_pre,
            postcondition=specification.postcondition,
            invariants=specification.invariants,
            name=f"{specification.name}_repaired",
            description="Precondition strengthened to exclude counterexample",
        )
    
    def _weaken_postcondition(
        self,
        program: QuantumProgram,
        specification: Specification,
        counterexample: CounterExample,
    ) -> Optional[Specification]:
        """Weaken postcondition to allow counterexample output."""
        # Create pattern that matches counterexample output
        allowance = self._create_output_allowance(counterexample)
        
        if allowance is None:
            return None
        
        # Disjoin with existing postcondition
        old_post = specification.postcondition.condition
        new_post_condition = CompoundCondition(
            operator="or",
            operands=[old_post, allowance]
        )
        
        new_post = Postcondition(
            condition=new_post_condition,
            description=f"Weakened: {specification.postcondition.description}"
        )
        
        return Specification(
            precondition=specification.precondition,
            postcondition=new_post,
            invariants=specification.invariants,
            name=f"{specification.name}_repaired",
            description="Postcondition weakened to allow counterexample output",
        )
    
    def _create_input_exclusion(
        self,
        counterexample: CounterExample,
    ) -> Optional[AtomicCondition]:
        """Create condition that excludes the counterexample input."""
        # Simplified: exclude the specific input pattern
        # In practice, would create more general exclusion
        
        input_state = counterexample.input_state
        if hasattr(input_state, 'num_qubits'):
            # Create predicate excluding this state pattern
            return AtomicCondition(
                predicate="not_state_pattern",
                arguments=[str(input_state)]
            )
        
        return None
    
    def _create_output_allowance(
        self,
        counterexample: CounterExample,
    ) -> Optional[AtomicCondition]:
        """Create condition that allows the counterexample output."""
        output_state = counterexample.output_state
        if hasattr(output_state, 'num_qubits'):
            # Create predicate allowing this state pattern
            return AtomicCondition(
                predicate="allows_state_pattern",
                arguments=[str(output_state)]
            )
        
        return None
    
    def _llm_repair(
        self,
        program: QuantumProgram,
        specification: Specification,
        counterexample: CounterExample,
        diagnosis: FailureDiagnosis,
    ) -> Optional[Specification]:
        """Use LLM to repair specification."""
        if self.llm is None:
            return None
        
        prompt = REPAIR_PROMPT.format(
            program_source=program.source[:1000],
            precondition=specification.precondition.to_human_readable(),
            postcondition=specification.postcondition.to_human_readable(),
            cex_input=str(counterexample.input_state),
            cex_output=str(counterexample.output_state),
            violated_condition=counterexample.violated_condition,
            diagnosis=diagnosis.name,
        )
        
        try:
            responses = self.llm.generate(prompt, n=1)
            if responses:
                response = responses[0]
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    data = json.loads(response[json_start:json_end])
                    
                    new_pre = Precondition.from_string(
                        data.get("precondition", "true")
                    )
                    new_post = Postcondition.from_string(
                        data.get("postcondition", "true")
                    )
                    
                    return Specification(
                        precondition=new_pre,
                        postcondition=new_post,
                        invariants=specification.invariants,
                        name=f"{specification.name}_llm_repaired",
                        description=data.get("reasoning", "LLM repair"),
                    )
        except Exception as e:
            logger.debug(f"LLM repair failed: {e}")
        
        return None


def repair_with_counterexample(
    program: QuantumProgram,
    specification: Specification,
    counterexample: CounterExample,
    llm: Optional[LLMInterface] = None,
) -> Optional[Specification]:
    """
    Convenience function to repair a specification.
    
    Args:
        program: The quantum program
        specification: Failed specification
        counterexample: Counterexample from verifier
        llm: Optional LLM interface for complex repairs
        
    Returns:
        Repaired specification or None
    """
    repairer = RepairSpecification(llm=llm)
    return repairer.repair(program, specification, counterexample)
