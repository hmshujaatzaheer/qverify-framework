#!/usr/bin/env python3
"""
QVERIFY LLM Evaluation Script

Evaluate different LLM models on quantum program verification tasks.

Usage:
    python scripts/evaluate_llm.py --models claude-3.5-sonnet gpt-4o --tier T2
    python scripts/evaluate_llm.py --all-models --output results/comparison.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models on QVERIFY benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["claude-3.5-sonnet"],
        help="LLM models to evaluate"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Evaluate all supported models"
    )
    
    parser.add_argument(
        "--tier",
        type=str,
        choices=["T1", "T2", "T3", "T4", "T5"],
        help="Benchmark tier to use"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout per program (seconds)"
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
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Define available models
    available_models = [
        "claude-3.5-sonnet",
        "claude-3-opus",
        "gpt-4o",
        "gpt-4-turbo",
        "llama-3-70b",
    ]
    
    if args.all_models:
        models = available_models
    else:
        models = args.models
    
    print("=" * 60)
    print("QVERIFY LLM EVALUATION")
    print("=" * 60)
    print(f"Models: {', '.join(models)}")
    print(f"Tier: {args.tier or 'all'}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 60)
    
    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "tier": args.tier,
            "timeout": args.timeout,
        },
        "models": {},
    }
    
    from qverify.benchmark import QVerifyBench
    
    for model in models:
        print(f"\n{'─' * 40}")
        print(f"Evaluating: {model}")
        print(f"{'─' * 40}")
        
        try:
            bench = QVerifyBench(tier=args.tier)
            results = bench.evaluate(
                llm=model,
                timeout=args.timeout,
                verbose=args.verbose,
            )
            
            all_results["models"][model] = results.to_dict()
            
            print(f"\nResults for {model}:")
            print(f"  Synthesis Rate:    {results.synthesis_rate:.2%}")
            print(f"  Verification Rate: {results.verification_rate:.2%}")
            print(f"  Avg Time:          {results.avg_time:.2f}s")
            
        except Exception as e:
            print(f"Error evaluating {model}: {e}")
            all_results["models"][model] = {"error": str(e)}
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Model':<25} {'Synth %':>10} {'Verify %':>10} {'Time':>10}")
    print("-" * 60)
    
    for model, results in all_results["models"].items():
        if "error" in results:
            print(f"{model:<25} {'ERROR':>10} {'─':>10} {'─':>10}")
        else:
            synth = results.get("synthesis_rate", 0) * 100
            verify = results.get("verification_rate", 0) * 100
            time_s = results.get("avg_time", 0)
            print(f"{model:<25} {synth:>9.1f}% {verify:>9.1f}% {time_s:>9.2f}s")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
