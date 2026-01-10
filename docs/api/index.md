# QVERIFY API Reference

## Overview

This documentation provides a complete reference for the QVERIFY Python API.

## Core Modules

### qverify.core

- **QuantumProgram** - Quantum program representation
- **Specification** - Formal specification types
- **QuantumStatePredicate** - Quantum state predicates
- **Types** - Core type definitions

### qverify.algorithms

- **QuantumSpecSynth** - LLM-guided specification synthesis
- **LearnQuantumPredicate** - Quantum predicate learning
- **SynthesizeQuantumInvariant** - Loop invariant synthesis
- **RepairSpecification** - Counterexample-guided repair

### qverify.verification

- **NeuralSilVer** - Main verification engine
- **VCGenerator** - Verification condition generation
- **SMTSolver** - SMT solver interface
- **CounterexampleAnalyzer** - Counterexample analysis

### qverify.benchmark

- **QVerifyBench** - Benchmark framework
- **Metrics** - Evaluation metrics

### qverify.utils

- **LLMInterface** - LLM provider interfaces
- **Parsers** - Quantum program parsers
- **Logging** - Logging utilities

## Quick Reference

```python
from qverify import QVerify, QuantumProgram

# Initialize
qv = QVerify(llm="claude-3.5-sonnet")

# Load program
program = QuantumProgram.from_silq("def f(q: qubit) { ... }")

# Synthesize and verify
spec, result = qv.synthesize_and_verify(program)
```

## See Also

- [Getting Started Tutorial](../tutorials/getting_started.md)
- [Examples](../../data/examples/)
- [GitHub Repository](https://github.com/hmshujaatzaheer/qverify-framework)
