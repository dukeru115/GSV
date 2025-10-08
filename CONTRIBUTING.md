# Contributing to GSV 2.0

Thank you for your interest in contributing to the Global State Vector 2.0 framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas for Contribution](#areas-for-contribution)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GSV.git
   cd GSV
   ```
3. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8 mypy
```

### Verify Installation

```bash
# Run tests to ensure everything works
python gsv2_tests.py
```

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to all classes and functions
- Keep functions focused and modular
- Maximum line length: 100 characters

### Code Formatting

We recommend using `black` for automatic formatting:

```bash
black *.py
```

### Type Hints

Use type hints where possible to improve code clarity:

```python
def compute_stress(self, td_error: float) -> float:
    """Compute stress from TD error"""
    ...
```

### Documentation

- Add clear docstrings following NumPy/Google style
- Update README.md if adding new features
- Include usage examples for new functionality
- Document mathematical formulations where relevant

## Testing

### Running Tests

```bash
# Run all tests
python gsv2_tests.py

# Run specific demo
python gsv2_gridworld_demo.py
```

### Writing Tests

When adding new features:

1. Add unit tests to `gsv2_tests.py`
2. Follow existing test structure
3. Ensure tests pass before submitting PR
4. Aim for high test coverage

Example test structure:

```python
def test_new_feature(self) -> bool:
    """Test description"""
    try:
        # Setup
        gsv = GSV2Controller()
        
        # Execute
        result = gsv.new_feature()
        
        # Verify
        assert result == expected_value, "Error message"
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False
```

## Pull Request Process

1. **Update Documentation**: Ensure README.md and docstrings are updated
2. **Run Tests**: Verify all tests pass
3. **Clean Commits**: Make clear, atomic commits with descriptive messages
4. **Update CHANGELOG**: Add entry describing your changes (if applicable)
5. **Submit PR**: Create pull request with clear description of changes

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] No breaking changes (or documented if necessary)
```

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority

- [ ] **Additional Agent Integrations**: DQN, PPO, A3C wrappers
- [ ] **OpenAI Gym Integration**: Wrapper for standard RL environments
- [ ] **LLM-Specific Metrics**: Metrics for language model agents
- [ ] **Empirical Validation**: Benchmarks on complex tasks (Atari, MuJoCo)

### Medium Priority

- [ ] **Multi-Agent Support**: Activate and test SS (Social) axis
- [ ] **Visualization Tools**: Interactive dashboards, real-time monitoring
- [ ] **Performance Optimization**: JAX/GPU acceleration
- [ ] **Additional Examples**: More domain-specific examples

### Research Contributions

- [ ] **Theoretical Analysis**: Formal derivations from Free Energy Principle
- [ ] **Parameter Tuning**: Automated hyperparameter optimization
- [ ] **Ablation Studies**: Component importance analysis
- [ ] **Comparison Studies**: Benchmarks vs. meta-learning approaches

## Code Review Process

1. Maintainers will review PRs within 1-2 weeks
2. Address review comments promptly
3. Keep PRs focused and reasonably sized
4. Be open to feedback and suggestions

## Community Guidelines

- Be respectful and constructive
- Help others learn and grow
- Focus on technical merit
- Give credit where due

## Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Check existing issues and documentation first
- Contact maintainers:
  - Timur Urmanov: urmanov.t@gmail.com
  - Kamil Gadeev: gadeev.kamil@gmail.com
  - Bakhtier Iusupov: usupovbahtiayr@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GSV 2.0!** 🚀
