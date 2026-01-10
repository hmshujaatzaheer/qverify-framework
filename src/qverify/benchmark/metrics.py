"""
Evaluation Metrics for QVerifyBench.

This module provides metrics for evaluating LLM performance on
quantum program verification tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import statistics


@dataclass
class SpecificationMetrics:
    """Metrics for specification quality."""
    
    syntactic_validity: float  # Parses correctly (0-1)
    semantic_completeness: float  # Covers all behaviors (0-1)
    ground_truth_similarity: float  # Similarity to reference (0-1)
    verification_rate: float  # Successfully verified (0-1)
    
    @property
    def overall_score(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "syntactic_validity": 0.2,
            "semantic_completeness": 0.3,
            "ground_truth_similarity": 0.2,
            "verification_rate": 0.3,
        }
        return (
            weights["syntactic_validity"] * self.syntactic_validity +
            weights["semantic_completeness"] * self.semantic_completeness +
            weights["ground_truth_similarity"] * self.ground_truth_similarity +
            weights["verification_rate"] * self.verification_rate
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for synthesis/verification."""
    
    synthesis_time_mean: float  # Average synthesis time (seconds)
    synthesis_time_std: float  # Std dev of synthesis time
    verification_time_mean: float  # Average verification time
    verification_time_std: float
    llm_calls_mean: float  # Average LLM API calls
    refinement_iterations_mean: float  # Average CEGIS iterations
    
    @classmethod
    def from_times(
        cls,
        synthesis_times: list[float],
        verification_times: list[float],
        llm_calls: list[int],
        refinement_iterations: list[int],
    ) -> 'PerformanceMetrics':
        """Create metrics from raw measurements."""
        return cls(
            synthesis_time_mean=statistics.mean(synthesis_times) if synthesis_times else 0,
            synthesis_time_std=statistics.stdev(synthesis_times) if len(synthesis_times) > 1 else 0,
            verification_time_mean=statistics.mean(verification_times) if verification_times else 0,
            verification_time_std=statistics.stdev(verification_times) if len(verification_times) > 1 else 0,
            llm_calls_mean=statistics.mean(llm_calls) if llm_calls else 0,
            refinement_iterations_mean=statistics.mean(refinement_iterations) if refinement_iterations else 0,
        )


@dataclass
class TierMetrics:
    """Metrics for a specific benchmark tier."""
    
    tier: str
    num_programs: int
    synthesis_success_rate: float
    verification_success_rate: float
    specification_metrics: SpecificationMetrics
    performance_metrics: PerformanceMetrics
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier,
            "num_programs": self.num_programs,
            "synthesis_success_rate": self.synthesis_success_rate,
            "verification_success_rate": self.verification_success_rate,
            "specification_metrics": {
                "syntactic_validity": self.specification_metrics.syntactic_validity,
                "semantic_completeness": self.specification_metrics.semantic_completeness,
                "ground_truth_similarity": self.specification_metrics.ground_truth_similarity,
                "verification_rate": self.specification_metrics.verification_rate,
                "overall_score": self.specification_metrics.overall_score,
            },
            "performance_metrics": {
                "synthesis_time_mean": self.performance_metrics.synthesis_time_mean,
                "synthesis_time_std": self.performance_metrics.synthesis_time_std,
                "verification_time_mean": self.performance_metrics.verification_time_mean,
                "verification_time_std": self.performance_metrics.verification_time_std,
                "llm_calls_mean": self.performance_metrics.llm_calls_mean,
                "refinement_iterations_mean": self.performance_metrics.refinement_iterations_mean,
            },
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report for a model."""
    
    model_name: str
    model_version: str
    timestamp: str
    tier_metrics: dict[str, TierMetrics] = field(default_factory=dict)
    overall_metrics: Optional[TierMetrics] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def add_tier_metrics(self, tier: str, metrics: TierMetrics) -> None:
        """Add metrics for a tier."""
        self.tier_metrics[tier] = metrics
    
    def compute_overall(self) -> None:
        """Compute overall metrics across all tiers."""
        if not self.tier_metrics:
            return
        
        total_programs = sum(m.num_programs for m in self.tier_metrics.values())
        
        # Weighted averages by number of programs
        weighted_synth_rate = sum(
            m.synthesis_success_rate * m.num_programs
            for m in self.tier_metrics.values()
        ) / total_programs
        
        weighted_verify_rate = sum(
            m.verification_success_rate * m.num_programs
            for m in self.tier_metrics.values()
        ) / total_programs
        
        # Average specification metrics
        avg_spec_metrics = SpecificationMetrics(
            syntactic_validity=statistics.mean(
                m.specification_metrics.syntactic_validity
                for m in self.tier_metrics.values()
            ),
            semantic_completeness=statistics.mean(
                m.specification_metrics.semantic_completeness
                for m in self.tier_metrics.values()
            ),
            ground_truth_similarity=statistics.mean(
                m.specification_metrics.ground_truth_similarity
                for m in self.tier_metrics.values()
            ),
            verification_rate=statistics.mean(
                m.specification_metrics.verification_rate
                for m in self.tier_metrics.values()
            ),
        )
        
        # Average performance metrics
        avg_perf_metrics = PerformanceMetrics(
            synthesis_time_mean=statistics.mean(
                m.performance_metrics.synthesis_time_mean
                for m in self.tier_metrics.values()
            ),
            synthesis_time_std=statistics.mean(
                m.performance_metrics.synthesis_time_std
                for m in self.tier_metrics.values()
            ),
            verification_time_mean=statistics.mean(
                m.performance_metrics.verification_time_mean
                for m in self.tier_metrics.values()
            ),
            verification_time_std=statistics.mean(
                m.performance_metrics.verification_time_std
                for m in self.tier_metrics.values()
            ),
            llm_calls_mean=statistics.mean(
                m.performance_metrics.llm_calls_mean
                for m in self.tier_metrics.values()
            ),
            refinement_iterations_mean=statistics.mean(
                m.performance_metrics.refinement_iterations_mean
                for m in self.tier_metrics.values()
            ),
        )
        
        self.overall_metrics = TierMetrics(
            tier="overall",
            num_programs=total_programs,
            synthesis_success_rate=weighted_synth_rate,
            verification_success_rate=weighted_verify_rate,
            specification_metrics=avg_spec_metrics,
            performance_metrics=avg_perf_metrics,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "tier_metrics": {
                tier: metrics.to_dict()
                for tier, metrics in self.tier_metrics.items()
            },
            "overall_metrics": self.overall_metrics.to_dict() if self.overall_metrics else None,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'EvaluationReport':
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        report = cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )
        
        # Reconstruct tier metrics
        for tier, metrics_data in data.get("tier_metrics", {}).items():
            spec_data = metrics_data["specification_metrics"]
            perf_data = metrics_data["performance_metrics"]
            
            report.tier_metrics[tier] = TierMetrics(
                tier=metrics_data["tier"],
                num_programs=metrics_data["num_programs"],
                synthesis_success_rate=metrics_data["synthesis_success_rate"],
                verification_success_rate=metrics_data["verification_success_rate"],
                specification_metrics=SpecificationMetrics(
                    syntactic_validity=spec_data["syntactic_validity"],
                    semantic_completeness=spec_data["semantic_completeness"],
                    ground_truth_similarity=spec_data["ground_truth_similarity"],
                    verification_rate=spec_data["verification_rate"],
                ),
                performance_metrics=PerformanceMetrics(
                    synthesis_time_mean=perf_data["synthesis_time_mean"],
                    synthesis_time_std=perf_data["synthesis_time_std"],
                    verification_time_mean=perf_data["verification_time_mean"],
                    verification_time_std=perf_data["verification_time_std"],
                    llm_calls_mean=perf_data["llm_calls_mean"],
                    refinement_iterations_mean=perf_data["refinement_iterations_mean"],
                ),
            )
        
        return report


def compute_specification_similarity(
    spec1_pre: str,
    spec1_post: str,
    spec2_pre: str,
    spec2_post: str,
) -> float:
    """
    Compute similarity between two specifications.
    
    Uses token-based Jaccard similarity.
    """
    def tokenize(s: str) -> set[str]:
        import re
        return set(re.findall(r'\w+', s.lower()))
    
    pre_tokens1 = tokenize(spec1_pre)
    pre_tokens2 = tokenize(spec2_pre)
    post_tokens1 = tokenize(spec1_post)
    post_tokens2 = tokenize(spec2_post)
    
    def jaccard(s1: set, s2: set) -> float:
        if not s1 and not s2:
            return 1.0
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union if union > 0 else 0.0
    
    pre_sim = jaccard(pre_tokens1, pre_tokens2)
    post_sim = jaccard(post_tokens1, post_tokens2)
    
    return (pre_sim + post_sim) / 2
