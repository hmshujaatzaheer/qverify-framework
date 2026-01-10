#!/usr/bin/env python3
"""
QVERIFY Benchmark Runner

Run QVerifyBench evaluations on different LLM models.

Usage:
    python scripts/run_benchmark.py --llm claude-3.5-sonnet --tier T2
    python scripts/run_benchmark.py --llm gpt-4o --all-tiers --output results.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qverify.benchmark import QVerifyBench, BenchmarkTier


def main():
    parser = argparse.ArgumentParser(
        description="Run QVERIFY benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --llm claude-3.5-sonnet --tier T2
  %(prog)s --llm gpt-4o --all-tiers
  %(prog)s --llm claude-3.5-sonnet --tier T1 --output results/t1_claude.json
        """
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        default="claude-3.5-sonnet",
        help="LLM model to evaluate (default: claude-3.5-sonnet)"
    )
    
    parser.add_argument(
        "--tier",
        type=str,
        choices=["T1", "T2", "T3", "T4", "T5"],
        help="Benchmark tier to run (T1-T5)"
    )
    
    parser.add_argument(
        "--all-tiers",
        action="store_true",
        help="Run all benchmark tiers"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per program in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Determine tiers to run
    if args.all_tiers:
        tiers = ["T1", "T2", "T3", "T4", "T5"]
    elif args.tier:
        tiers = [args.tier]
    else:
        tiers = [None]  # All tiers combined
    
    all_results = []
    
    for tier in tiers:
        print(f"\n{'='*60}")
        print(f"Running benchmark: tier={tier or 'all'}, llm={args.llm}")
        print(f"{'='*60}\n")
        
        # Initialize benchmark
        bench = QVerifyBench(tier=tier)
        print(f"Loaded {len(bench)} programs")
        
        # Run evaluation
        results = bench.evaluate(
            llm=args.llm,
            timeout=args.timeout,
            verbose=args.verbose,
        )
        
        all_results.append(results)
        
        # Print summary
        print(f"\n{'─'*40}")
        print(f"Results for tier {tier or 'all'}:")
        print(f"  Synthesis Rate:    {results.synthesis_rate:.2%}")
        print(f"  Verification Rate: {results.verification_rate:.2%}")
        print(f"  Avg Time:          {results.avg_time:.2f}s")
        print(f"  Avg Spec Match:    {results.avg_spec_match:.2%}")
        print(f"{'─'*40}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "llm": args.llm,
            "timeout": args.timeout,
            "results": [r.to_dict() for r in all_results],
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_programs = sum(r.total_programs for r in all_results)
    total_synth = sum(sum(1 for e in r.results if e.synthesis_success) for r in all_results)
    total_verify = sum(sum(1 for e in r.results if e.verification_success) for r in all_results)
    
    print(f"Total Programs:      {total_programs}")
    print(f"Synthesis Success:   {total_synth} ({total_synth/total_programs:.2%})")
    print(f"Verification Success: {total_verify} ({total_verify/total_programs:.2%})")


if __name__ == "__main__":
    main()
