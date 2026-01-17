# QVERIFY: LLM-Assisted Formal Verification of Quantum Programs

<p align="center">
  <a href="https://github.com/hmshujaatzaheer/qverify-framework/actions/workflows/ci.yml"><img src="https://github.com/hmshujaatzaheer/qverify-framework/actions/workflows/ci.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/hmshujaatzaheer/qverify-framework"><img src="https://codecov.io/gh/hmshujaatzaheer/qverify-framework/graph/badge.svg?token=fbbe4b85-3b5d-4938-93a6-696efab91a3f" alt="codecov"></a>
  <a href="https://github.com/hmshujaatzaheer/qverify-framework/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python"></a>
</p>

<p align="center">
  <b>Bridging Neural Reasoning and Quantum Correctness</b>
</p>

---

## Overview

**QVERIFY** is a framework that integrates Large Language Model (LLM) reasoning capabilities with formal verification of quantum programs. As LLMs increasingly generate quantum circuits, the need for formal correctness guarantees becomes critical.

QVERIFY addresses three key challenges:
1. **Specification Synthesis**: Automatically generate formal specifications for quantum programs using LLMs
2. **Verification Integration**: Connect LLM-synthesized specs with SMT-based quantum verification
3. **Benchmark Evaluation**: Evaluate LLM capabilities on quantum program verification tasks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QVERIFY Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────────┐     │
│  │ Quantum       │    │ QuantumSpecSynth │    │ NeuralVerifier       │     │
│  │ Program       │───▶│ (LLM Module)     │───▶│ (SMT + LLM)          │     │
│  │ (Silq/QASM)   │    │                  │    │                      │     │
│  └───────────────┘    └──────────────────┘    └──────────────────────┘     │
│         │                     │                        │                    │
│         │                     ▼                        ▼                    │
│         │            ┌──────────────────┐    ┌──────────────────────┐     │
│         │            │ Specification    │    │ Verification         │     │
│         │            │ (Pre, Post, Inv) │    │ Result               │     │
│         │            └──────────────────┘    └──────────────────────┘     │
│         │                     ▲                        │                    │
│         │                     │    Counterexample      │                    │
│         │                     └────────────────────────┘                    │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                      QVerifyBench (500+ Programs)                  │     │
│  │  T1: Basic (150) │ T2: Intermediate (150) │ T3: Standard (100)   │     │
│  │  T4: Advanced (75) │ T5: Research (25)                            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **QuantumSpecSynth**: LLM-guided specification synthesis with quantum-aware prompting
- **LearnQuantumPredicate**: Neural predicate synthesis for quantum states
- **SynthesizeQuantumInvariant**: Loop invariant generation for iterative quantum algorithms
- **RepairSpecification**: Counterexample-guided specification refinement
- **NeuralVerifier**: Integrated verification combining LLM reasoning with SMT solving
- **QVerifyBench**: 500+ quantum programs benchmark with ground-truth specifications
- **Multi-LLM Support**: Claude, GPT-4, Llama, and custom fine-tuned models

## Quick Start

### Installation

```bash
git clone https://github.com/hmshujaatzaheer/qverify-framework.git
cd qverify-framework
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### Basic Usage

```python
from qverify import QVerify, QuantumProgram

# Initialize QVERIFY with your preferred LLM
qv = QVerify(llm="claude-3.5-sonnet", backend="z3")

# Load a quantum program
program = QuantumProgram.from_silq("""
def grover_iteration(qubits: qubit[], oracle: qubit[] -> qubit[]) {
    qubits = hadamard(qubits);
    qubits = oracle(qubits);
    qubits = hadamard(qubits);
    qubits = phase_flip(qubits);
    qubits = hadamard(qubits);
    return qubits;
}
""")

# Synthesize specification using LLM
spec = qv.synthesize_specification(program)
print(f"Precondition: {spec.precondition}")
print(f"Postcondition: {spec.postcondition}")

# Verify the program
result = qv.verify(program, spec)
print(f"Verification Result: {result.status}")
```

### Running QVerifyBench

```python
from qverify.benchmark import QVerifyBench

# Load benchmark
bench = QVerifyBench(tier="T2")

# Evaluate an LLM
results = bench.evaluate(llm="gpt-4o", timeout=60)

print(f"Synthesis Success Rate: {results.synthesis_rate:.2%}")
print(f"Verification Success Rate: {results.verification_rate:.2%}")
```

## Project Structure

```
qverify-framework/
├── src/qverify/
│   ├── core/                 # Core data structures
│   ├── algorithms/           # Synthesis algorithms (5 algorithms)
│   ├── verification/         # Verification engine
│   ├── benchmark/            # QVerifyBench (500+ programs)
│   └── utils/                # Utilities
├── tests/                    # Test suite
├── docs/                     # Documentation
└── configs/                  # Configuration files
```

## Core Algorithms

| Algorithm | Description |
|-----------|-------------|
| QuantumSpecSynth | LLM-guided specification synthesis with quantum-aware prompting |
| LearnQuantumPredicate | Neural predicate synthesis for quantum states |
| SynthesizeQuantumInvariant | Loop invariant generation for iterative algorithms |
| RepairSpecification | Counterexample-guided specification refinement |
| NeuralVerifier | SMT-based verification with LLM-assisted lemma synthesis |

## QVerifyBench

A comprehensive benchmark for quantum program verification with 500+ programs:

| Tier | Programs | Complexity | Examples |
|------|----------|------------|----------|
| T1 | 150 | Single qubit, no loops | Hadamard, Pauli gates |
| T2 | 150 | Multi-qubit, entanglement | Bell states, GHZ |
| T3 | 100 | Loops, oracles | Grover, QFT |
| T4 | 75 | Error correction | Steane, Surface code |
| T5 | 25 | Research-level | VQE, QAOA |

## Configuration

```yaml
qverify:
  llm:
    provider: "anthropic"
    model: "claude-3.5-sonnet"
    temperature: 0.2
  verification:
    solver: "z3"
    timeout: 30
```

## Testing

```bash
pytest tests/ -v --cov=src/qverify
```

## Citation

```bibtex
@software{qverify2026,
  title={QVERIFY: LLM-Assisted Formal Verification of Quantum Programs},
  author={Zaheer, H M Shujaat},
  year={2026},
  url={https://github.com/hmshujaatzaheer/qverify-framework}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: H M Shujaat Zaheer
- **Email**: shujabis@gmail.com
- **GitHub**: [@hmshujaatzaheer](https://github.com/hmshujaatzaheer)
