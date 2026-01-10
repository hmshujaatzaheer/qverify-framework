#!/usr/bin/env python3
"""
QVERIFY Batch Specification Generation

Generate specifications for multiple quantum programs in batch.

Usage:
    python scripts/generate_specs.py --input programs/ --output specs/
    python scripts/generate_specs.py --file program.silq --llm gpt-4o
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def process_file(
    filepath: Path,
    qv: Any,
    verify: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Process a single file and generate specification."""
    from qverify import QuantumProgram
    from qverify.core.types import SynthesisStatus, VerificationStatus
    
    result = {
        "file": str(filepath),
        "name": filepath.stem,
        "status": "unknown",
        "specification": None,
        "verified": False,
        "error": None,
    }
    
    try:
        # Load program
        program = QuantumProgram.from_file(filepath)
        
        if verbose:
            print(f"  Loaded: {program.num_qubits} qubits, {program.num_gates} gates")
        
        # Synthesize specification
        synth_result = qv.synthesize_specification(program, verify=verify)
        
        if synth_result.status == SynthesisStatus.SUCCESS:
            result["status"] = "success"
            result["specification"] = synth_result.specification.to_dict()
            result["synthesis_time"] = synth_result.time_seconds
            result["llm_calls"] = synth_result.llm_calls
            
            if verify:
                verify_result = qv.verify(program, synth_result.specification)
                result["verified"] = verify_result.status == VerificationStatus.VALID
                result["verification_time"] = verify_result.time_seconds
        else:
            result["status"] = "failed"
            result["error"] = synth_result.message
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate specifications for quantum programs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input directory containing quantum programs"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Single file to process"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output/specs",
        help="Output directory for specifications"
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        default="claude-3.5-sonnet",
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification of generated specs"
    )
    
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".silq", ".qasm", ".sq"],
        help="File extensions to process"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if not args.input and not args.file:
        parser.error("Either --input or --file must be specified")
    
    from qverify import QVerify
    
    print("=" * 60)
    print("QVERIFY BATCH SPECIFICATION GENERATION")
    print("=" * 60)
    print(f"LLM: {args.llm}")
    print(f"Verify: {not args.no_verify}")
    print("=" * 60)
    
    # Initialize QVerify
    qv = QVerify(llm=args.llm)
    
    # Collect files to process
    files_to_process = []
    
    if args.file:
        files_to_process.append(Path(args.file))
    
    if args.input:
        input_dir = Path(args.input)
        for ext in args.extensions:
            files_to_process.extend(input_dir.glob(f"*{ext}"))
            files_to_process.extend(input_dir.glob(f"**/*{ext}"))
    
    # Remove duplicates
    files_to_process = list(set(files_to_process))
    
    print(f"\nFound {len(files_to_process)} files to process")
    
    # Process files
    results = []
    success_count = 0
    verified_count = 0
    
    for i, filepath in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Processing: {filepath.name}")
        
        result = process_file(
            filepath,
            qv,
            verify=not args.no_verify,
            verbose=args.verbose,
        )
        
        results.append(result)
        
        if result["status"] == "success":
            success_count += 1
            status_str = "✓ Success"
            if result.get("verified"):
                verified_count += 1
                status_str += " (verified)"
        elif result["status"] == "failed":
            status_str = "✗ Failed"
        else:
            status_str = "⚠ Error"
        
        print(f"  Status: {status_str}")
        
        if result.get("error") and args.verbose:
            print(f"  Error: {result['error']}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual specifications
    for result in results:
        if result["specification"]:
            spec_file = output_dir / f"{result['name']}_spec.json"
            with open(spec_file, 'w') as f:
                json.dump(result["specification"], f, indent=2)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "llm": args.llm,
            "verify": not args.no_verify,
        },
        "summary": {
            "total": len(results),
            "success": success_count,
            "verified": verified_count,
            "failed": len(results) - success_count,
        },
        "results": results,
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files:      {len(results)}")
    print(f"Successful:       {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Verified:         {verified_count} ({verified_count/len(results)*100:.1f}%)")
    print(f"Failed/Error:     {len(results) - success_count}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
