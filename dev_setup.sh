#!/bin/bash
# Development Environment Setup Script for GSV 2.0
# This script sets up a complete development environment

set -e  # Exit on error

echo "=================================================="
echo "GSV 2.0 - Development Environment Setup"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv gsv_env
    
    # Activate based on OS
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        source gsv_env/Scripts/activate
    else
        source gsv_env/bin/activate
    fi
    
    echo "✓ Virtual environment created and activated"
    echo ""
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt
echo "✓ Core dependencies installed"
echo ""

# Install development dependencies
read -p "Install development tools? (pytest, black, flake8, mypy) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing development tools..."
    pip install pytest pytest-cov black flake8 mypy
    echo "✓ Development tools installed"
    echo ""
fi

# Run installation check
echo "Verifying installation..."
python3 check_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Development environment setup complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Run tests: python gsv2_tests.py"
    echo "  2. Try examples: python examples/simple_example.py"
    echo "  3. Read CONTRIBUTING.md for guidelines"
    echo "  4. Start coding!"
    echo ""
    echo "If you created a virtual environment, activate it with:"
    echo "  source gsv_env/bin/activate  (Linux/macOS)"
    echo "  gsv_env\\Scripts\\activate    (Windows)"
    echo ""
else
    echo ""
    echo "❌ Setup verification failed"
    echo "Please check error messages above"
    exit 1
fi
