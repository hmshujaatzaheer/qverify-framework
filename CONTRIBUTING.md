# Contributing to QVERIFY

Thank you for your interest in contributing to QVERIFY! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Contributions](#making-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to shujabis@gmail.com.

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Assume good intentions

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are actively seeking contributors
- Feel free to ask questions on any issue before starting work

## Development Setup

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/qverify-framework.git
   cd qverify-framework
   ```

3. **Set up upstream remote**

   ```bash
   git remote add upstream https://github.com/hmshujaatzaheer/qverify-framework.git
   ```

4. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

6. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

7. **Verify setup**

   ```bash
   pytest
   ```

## Making Contributions

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **Features**: Add new functionality
- **Documentation**: Improve docs, add examples
- **Tests**: Increase test coverage
- **Benchmarks**: Add new benchmark programs
- **Performance**: Optimize existing code

### Contribution Workflow

1. **Create a branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes**

   - Write clear, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests locally**

   ```bash
   pytest
   pytest --cov=qverify --cov-report=html
   ```

4. **Run linters**

   ```bash
   ruff check src/
   mypy src/
   black --check src/
   ```

5. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add quantum predicate parsing"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` adding tests
   - `refactor:` code refactoring
   - `perf:` performance improvement

6. **Push and create PR**

   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Process

1. **Before submitting**
   - Ensure all tests pass
   - Update documentation if needed
   - Add entry to CHANGELOG.md

2. **PR Description**
   - Describe what changes you made
   - Link related issues
   - Include screenshots for UI changes

3. **Review process**
   - Maintainers will review your PR
   - Address any feedback
   - Once approved, your PR will be merged

4. **After merge**
   - Delete your branch
   - Sync your fork with upstream

## Coding Standards

### Python Style

We follow PEP 8 with these tools:

- **Black** for formatting (line length 100)
- **Ruff** for linting
- **isort** for import sorting
- **mypy** for type checking

### Code Guidelines

```python
# Use type hints
def synthesize_spec(program: QuantumProgram, timeout: float = 30.0) -> Specification:
    """
    Synthesize specification for quantum program.
    
    Args:
        program: The quantum program to analyze
        timeout: Maximum time in seconds
        
    Returns:
        Synthesized specification
        
    Raises:
        SynthesisError: If synthesis fails
    """
    ...

# Use dataclasses for data structures
@dataclass
class VerificationResult:
    status: VerificationStatus
    counterexample: Optional[CounterExample] = None
    time_seconds: float = 0.0

# Document public APIs with docstrings
# Use descriptive variable names
# Keep functions focused and small
```

### File Organization

```
src/qverify/
├── __init__.py          # Public API exports
├── core/                # Core data structures
├── algorithms/          # Main algorithms
├── verification/        # Verification infrastructure
├── benchmark/           # Benchmarking
└── utils/               # Utilities
```

## Testing Guidelines

### Writing Tests

```python
import pytest
from qverify import QVerify, QuantumProgram

class TestSpecificationSynthesis:
    """Tests for specification synthesis."""
    
    def test_hadamard_synthesis(self):
        """Test synthesis for Hadamard gate."""
        program = QuantumProgram.from_silq("def h(q: qubit) { q = H(q); return q; }")
        qv = QVerify(llm="mock")
        
        result = qv.synthesize_specification(program)
        
        assert result.status == SynthesisStatus.SUCCESS
        assert "superposition" in result.specification.postcondition.to_human_readable().lower()
    
    @pytest.mark.parametrize("gate", ["H", "X", "Y", "Z"])
    def test_single_qubit_gates(self, gate):
        """Test synthesis for various single-qubit gates."""
        ...
```

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Benchmark tests**: Verify benchmark functionality

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/unit/test_spec_synth.py

# With coverage
pytest --cov=qverify --cov-report=html

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def verify(self, program: QuantumProgram, spec: Specification) -> VerificationResult:
    """
    Verify a quantum program against its specification.
    
    This method uses SMT-based verification to check if the program
    satisfies the given specification.
    
    Args:
        program: The quantum program to verify
        spec: The specification to verify against
        
    Returns:
        VerificationResult containing status and optional counterexample
        
    Raises:
        VerificationError: If verification encounters an error
        
    Example:
        >>> qv = QVerify()
        >>> result = qv.verify(program, spec)
        >>> if result.is_valid():
        ...     print("Verified!")
    """
```

### Building Documentation

```bash
cd docs
make html
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email shujabis@gmail.com for private inquiries

Thank you for contributing to QVERIFY! 🎉
