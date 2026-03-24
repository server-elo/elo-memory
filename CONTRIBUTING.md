# Contributing to Elo Memory

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Contributors

- **Lorenc Ndoj** - Original author
- **Elvi Zekaj** - Core contributor

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/elo-memory.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Format code: `black src/`
5. Lint: `flake8 src/`
6. Commit: `git commit -m "Add feature: description"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- Follow PEP 8
- Use Black for formatting
- Add docstrings to all public functions
- Type hints encouraged

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% coverage

## Reporting Bugs

Please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/tracebacks

## Feature Requests

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Proposed API (if applicable)

## Code of Conduct

Be respectful, constructive, and inclusive.