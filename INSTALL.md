# Installation Guide for GSV 2.0

This guide provides detailed installation instructions for the Global State Vector 2.0 framework.

## Quick Install

```bash
# Clone the repository
git clone https://github.com/dukeru115/GSV.git
cd GSV

# Install dependencies
pip install -r requirements.txt

# Verify installation
python gsv2_tests.py
```

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, Linux
- **Dependencies**:
  - numpy >= 1.20.0
  - matplotlib >= 3.3.0
  - scipy >= 1.6.0

## Installation Methods

### Method 1: Using pip with requirements.txt (Recommended)

```bash
pip install -r requirements.txt
```

### Method 2: Using setup.py

```bash
# Development installation (editable)
pip install -e .

# Standard installation
pip install .
```

### Method 3: Manual dependency installation

```bash
pip install numpy>=1.20.0 matplotlib>=3.3.0 scipy>=1.6.0
```

## Verification

After installation, verify everything works:

```bash
# Run comprehensive test suite (should take ~30 seconds)
python gsv2_tests.py

# Expected output:
# ======================================================================
# GSV 2.0 CORE TESTS
# ======================================================================
# [✓] Params initialization
# [✓] Controller initialization
# ... (16 tests total)
# ======================================================================
# ✓ ALL TESTS PASSED (16/16)
```

## Running Examples

### Quick Start Example

```bash
python gsv2_quickstart.py
```

This will run through 6 practical examples demonstrating GSV usage.

### Gridworld Demo

```bash
python gsv2_gridworld_demo.py
```

This runs a complete 3000-episode experiment with regime changes (~5 minutes).

### Stress Response Demo

```bash
python gsv2_stress_demo.py
```

Demonstrates arousal response to periodic stress episodes (~1 minute).

## Development Installation

For development work, install additional tools:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Or use setup.py extras
pip install -e ".[dev]"
```

## Virtual Environment (Recommended)

Using a virtual environment is recommended to avoid dependency conflicts:

### Using venv

```bash
# Create virtual environment
python -m venv gsv_env

# Activate (Linux/macOS)
source gsv_env/bin/activate

# Activate (Windows)
gsv_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Create conda environment
conda create -n gsv python=3.10
conda activate gsv

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting

### Issue: ModuleNotFoundError for numpy/matplotlib/scipy

**Solution**: Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: Permission denied during installation

**Solution**: Use user installation:
```bash
pip install --user -r requirements.txt
```

### Issue: Tests fail with import errors

**Solution**: Ensure you're in the correct directory:
```bash
cd /path/to/GSV
python gsv2_tests.py
```

### Issue: Matplotlib backend errors

**Solution**: Set a non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
```

Or install a specific backend:
```bash
# For Linux
sudo apt-get install python3-tk

# For macOS
brew install python-tk
```

## Platform-Specific Notes

### Windows

- Use Command Prompt or PowerShell
- Paths use backslashes: `gsv_env\Scripts\activate`
- May need Visual C++ Build Tools for scipy

### macOS

- Python 3 command may be `python3` instead of `python`
- Use `pip3` instead of `pip` if needed
- Xcode Command Line Tools required for scipy

### Linux

- Typically works out of the box
- May need to install python3-dev: `sudo apt-get install python3-dev`
- For matplotlib GUI: `sudo apt-get install python3-tk`

## GPU Acceleration (Optional)

GSV 2.0 currently runs on CPU. For GPU acceleration in the future:

```bash
# For CUDA support (future feature)
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Uninstallation

```bash
# If installed with setup.py
pip uninstall gsv2

# Clean up virtual environment
deactivate
rm -rf gsv_env
```

## Next Steps

After successful installation:

1. **Run Tests**: `python gsv2_tests.py`
2. **Try Examples**: `python gsv2_quickstart.py`
3. **Read Documentation**: Check `README.md` and `Summary.md`
4. **Integrate**: Add GSV to your own agent

## Support

If you encounter issues:

1. Check this troubleshooting section
2. Verify Python version: `python --version`
3. Check installed packages: `pip list | grep -E "numpy|matplotlib|scipy"`
4. Create an issue on GitHub with:
   - Python version
   - Operating system
   - Error message
   - Steps to reproduce

## Contact

- Timur Urmanov: urmanov.t@gmail.com
- Kamil Gadeev: gadeev.kamil@gmail.com
- Bakhtier Iusupov: usupovbahtiayr@gmail.com

---

**Happy coding with GSV 2.0!** 🚀
