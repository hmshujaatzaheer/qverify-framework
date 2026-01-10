"""
QVerify: Main entry point for LLM-Assisted Quantum Program Verification.

This module provides the QVerify class which integrates specification synthesis,
verification, and repair into a unified interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from qverify.core.quantum_program import QuantumProgram
from qverify.core.specification import Specification
from qverify.core.types import (
    VerificationResult,
    VerificationStatus,
    SynthesisResult,
    SynthesisStatus,
)
from qverify.algorithms.spec_synth import QuantumSpecSynth
from qverify.algorithms.spec_repair import RepairSpecification
from qverify.verification.neural_silver import NeuralSilVer, create_z3_interface
from qverify.utils.llm_interface import create_llm_interface, LLMConfig


logger = logging.getLogger(__name__)


@dataclass
class QVerifyConfig:
    """Configuration for QVerify."""
    
    # LLM settings
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    
    # Verification settings
    smt_solver: str = "z3"
    verification_timeout: float = 30.0
    enable_lemma_hints: bool = True
    
    # Synthesis settings
    max_candidates: int = 5
    max_refinement_iterations: int = 3
    enable_quantum_aware_prompting: bool = True
    
    # General settings
    log_level: str = "INFO"
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'QVerifyConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data.get("qverify", {}))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "smt_solver": self.smt_solver,
            "verification_timeout": self.verification_timeout,
            "enable_lemma_hints": self.enable_lemma_hints,
            "max_candidates": self.max_candidates,
            "max_refinement_iterations": self.max_refinement_iterations,
            "enable_quantum_aware_prompting": self.enable_quantum_aware_prompting,
            "log_level": self.log_level,
            "cache_dir": self.cache_dir,
        }


@dataclass
class QVerify:
    """
    Main interface for LLM-assisted quantum program verification.
    
    QVerify integrates:
    - QuantumSpecSynth for specification synthesis
    - NeuralSilVer for verification
    - RepairSpecification for counterexample-guided refinement
    
    Example:
        >>> qv = QVerify(llm="claude-3.5-sonnet")
        >>> program = QuantumProgram.from_silq("def bell() { ... }")
        >>> spec = qv.synthesize_specification(program)
        >>> result = qv.verify(program, spec)
        >>> print(result.status)
    
    Args:
        llm: LLM model identifier (e.g., "claude-3.5-sonnet", "gpt-4o")
        backend: SMT solver backend ("z3" or "mock")
        config: Optional QVerifyConfig for detailed configuration
    """
    
    llm: str = "claude-3-5-sonnet-20241022"
    backend: str = "z3"
    config: Optional[QVerifyConfig] = None
    
    # Components (initialized lazily)
    _llm_interface: Any = field(default=None, init=False, repr=False)
    _spec_synth: Optional[QuantumSpecSynth] = field(default=None, init=False, repr=False)
    _verifier: Optional[NeuralSilVer] = field(default=None, init=False, repr=False)
    _repairer: Optional[RepairSpecification] = field(default=None, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize configuration."""
        if self.config is None:
            self.config = QVerifyConfig(
                llm_model=self.llm,
                smt_solver=self.backend,
            )
        
        # Set up logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
    
    def _ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if self._initialized:
            return
        
        # Initialize LLM interface
        try:
            self._llm_interface = create_llm_interface(LLMConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            ))
        except Exception as e:
            logger.warning(f"Could not initialize LLM: {e}. Running without LLM support.")
            self._llm_interface = None
        
        # Initialize SMT solver
        smt_solver = None
        if self.backend == "z3":
            smt_solver = create_z3_interface()
            if smt_solver is None:
                logger.warning("Z3 not available. Using mock verification.")
        
        # Initialize components
        self._spec_synth = QuantumSpecSynth(
            llm=self._llm_interface,
            max_candidates=self.config.max_candidates,
            max_refinement_iterations=self.config.max_refinement_iterations,
            enable_quantum_aware_prompting=self.config.enable_quantum_aware_prompting,
        )
        
        self._verifier = NeuralSilVer(
            smt_solver=smt_solver,
            llm=self._llm_interface,
            timeout=self.config.verification_timeout,
            enable_lemma_hints=self.config.enable_lemma_hints,
        )
        
        self._repairer = RepairSpecification(
            llm=self._llm_interface,
            max_repair_attempts=self.config.max_refinement_iterations,
        )
        
        self._initialized = True
    
    def synthesize_specification(
        self,
        program: QuantumProgram,
        verify: bool = True,
    ) -> SynthesisResult:
        """
        Synthesize a formal specification for a quantum program.
        
        Uses LLM-guided synthesis with optional verification and refinement.
        
        Args:
            program: The quantum program to synthesize specification for
            verify: Whether to verify the synthesized specification
            
        Returns:
            SynthesisResult containing the specification or failure info
        """
        self._ensure_initialized()
        
        verification_oracle = self._verifier if verify else None
        
        return self._spec_synth.synthesize(
            program,
            verification_oracle=verification_oracle,
            enable_refinement=verify,
        )
    
    def verify(
        self,
        program: QuantumProgram,
        specification: Specification,
    ) -> VerificationResult:
        """
        Verify a quantum program against its specification.
        
        Args:
            program: The quantum program to verify
            specification: The specification to verify against
            
        Returns:
            VerificationResult indicating success, failure, or unknown
        """
        self._ensure_initialized()
        
        return self._verifier.verify(program, specification)
    
    def synthesize_and_verify(
        self,
        program: QuantumProgram,
        max_attempts: int = 3,
    ) -> tuple[Optional[Specification], VerificationResult]:
        """
        Synthesize specification and verify in one step.
        
        Includes automatic repair on failure.
        
        Args:
            program: The quantum program
            max_attempts: Maximum synthesis/repair attempts
            
        Returns:
            Tuple of (specification, verification_result)
        """
        self._ensure_initialized()
        
        for attempt in range(max_attempts):
            # Synthesize
            synth_result = self.synthesize_specification(program, verify=True)
            
            if synth_result.status == SynthesisStatus.SUCCESS:
                spec = synth_result.specification
                
                # Verify
                verify_result = self.verify(program, spec)
                
                if verify_result.status == VerificationStatus.VALID:
                    return spec, verify_result
                
                # Try repair on failure
                if verify_result.counterexample is not None:
                    repaired = self._repairer.repair(
                        program, spec, verify_result.counterexample
                    )
                    if repaired is not None:
                        repair_result = self.verify(program, repaired)
                        if repair_result.status == VerificationStatus.VALID:
                            return repaired, repair_result
            
            logger.debug(f"Attempt {attempt + 1} failed, retrying...")
        
        # Return last attempt results
        return None, VerificationResult(
            status=VerificationStatus.UNKNOWN,
            message=f"Failed after {max_attempts} attempts",
        )
    
    def verify_from_source(
        self,
        source: str,
        language: str = "silq",
        specification: Optional[Specification] = None,
    ) -> VerificationResult:
        """
        Verify a program from source code string.
        
        Args:
            source: The program source code
            language: Programming language ("silq", "openqasm")
            specification: Optional specification (synthesized if not provided)
            
        Returns:
            VerificationResult
        """
        # Parse program
        if language.lower() == "silq":
            program = QuantumProgram.from_silq(source)
        elif language.lower() in {"openqasm", "qasm"}:
            program = QuantumProgram.from_openqasm(source)
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # Synthesize specification if not provided
        if specification is None:
            synth_result = self.synthesize_specification(program)
            if synth_result.status != SynthesisStatus.SUCCESS:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    message=f"Specification synthesis failed: {synth_result.message}",
                )
            specification = synth_result.specification
        
        return self.verify(program, specification)
    
    def verify_file(
        self,
        path: Union[str, Path],
        specification: Optional[Specification] = None,
    ) -> VerificationResult:
        """
        Verify a quantum program from file.
        
        Args:
            path: Path to the program file
            specification: Optional specification
            
        Returns:
            VerificationResult
        """
        program = QuantumProgram.from_file(path)
        
        if specification is None:
            synth_result = self.synthesize_specification(program)
            if synth_result.status != SynthesisStatus.SUCCESS:
                return VerificationResult(
                    status=VerificationStatus.UNKNOWN,
                    message=f"Specification synthesis failed: {synth_result.message}",
                )
            specification = synth_result.specification
        
        return self.verify(program, specification)
    
    def get_stats(self) -> dict[str, Any]:
        """Get verification statistics."""
        self._ensure_initialized()
        
        stats = {
            "llm_available": self._llm_interface is not None,
            "smt_available": self._verifier.smt_solver is not None,
        }
        
        if hasattr(self._verifier, '_stats'):
            stats["verification"] = self._verifier._get_stats_dict()
        
        return stats


def verify_quantum_program(
    source: str,
    language: str = "silq",
    llm: str = "claude-3.5-sonnet",
    backend: str = "z3",
) -> VerificationResult:
    """
    Convenience function to verify a quantum program.
    
    Args:
        source: Program source code
        language: Programming language
        llm: LLM model to use
        backend: SMT solver backend
        
    Returns:
        VerificationResult
    """
    qv = QVerify(llm=llm, backend=backend)
    return qv.verify_from_source(source, language)
