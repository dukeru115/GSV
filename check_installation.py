#!/usr/bin/env python3
"""
GSV 2.0 - Installation Verification Script

This script checks if your GSV 2.0 installation is working correctly.
Run this after installing dependencies to verify everything is set up properly.
"""

import sys
import importlib


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version >= (3, 8):
        print(f"   ✓ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_dependency(module_name, display_name=None):
    """Check if a dependency is installed"""
    if display_name is None:
        display_name = module_name
    
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"   ✓ {display_name} {version}")
        return True
    except ImportError:
        print(f"   ✗ {display_name} (not installed)")
        return False


def check_gsv_modules():
    """Check if GSV 2.0 modules are importable"""
    print("\n🔍 Checking GSV 2.0 modules...")
    
    modules = [
        ('gsv2_core', 'GSV2 Core'),
        ('gsv2_qlearning', 'GSV2 Q-Learning'),
        ('gsv2_analysis', 'GSV2 Analysis'),
    ]
    
    all_ok = True
    for module_name, display_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"   ✓ {display_name}")
        except ImportError as e:
            print(f"   ✗ {display_name} ({e})")
            all_ok = False
    
    return all_ok


def run_quick_test():
    """Run a quick functional test"""
    print("\n🔍 Running quick functional test...")
    
    try:
        from gsv2_core import GSV2Controller, GSV2Params, MetricComputer
        import numpy as np
        
        # Initialize
        gsv = GSV2Controller(GSV2Params.conservative())
        metrics = MetricComputer()
        
        # Run a few steps
        for _ in range(10):
            td_error = np.random.randn() * 0.1
            stress = metrics.compute_stress(td_error)
            
            gsv.step({
                'rho_def': stress,
                'R': 0.5,
                'novelty': 0.1,
                'SIR': 0.0,
                'F': 0.0
            })
        
        # Check state
        state = gsv.get_state()
        assert all(abs(state[k]) < 10 for k in state), "State diverged"
        
        print("   ✓ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False


def main():
    """Run all checks"""
    print("=" * 70)
    print("GSV 2.0 - Installation Verification")
    print("=" * 70)
    print()
    
    checks = []
    
    # Check Python version
    checks.append(check_python_version())
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    checks.append(check_dependency('numpy', 'NumPy'))
    checks.append(check_dependency('matplotlib', 'Matplotlib'))
    checks.append(check_dependency('scipy', 'SciPy'))
    
    # Check GSV modules
    checks.append(check_gsv_modules())
    
    # Run quick test
    checks.append(run_quick_test())
    
    # Summary
    print("\n" + "=" * 70)
    if all(checks):
        print("✅ ALL CHECKS PASSED")
        print("=" * 70)
        print()
        print("Your GSV 2.0 installation is working correctly!")
        print()
        print("Next steps:")
        print("  1. Run a simple example: python examples/simple_example.py")
        print("  2. Run the test suite: python gsv2_tests.py")
        print("  3. Try the gridworld demo: python gsv2_gridworld_demo.py")
        print("  4. Read QUICKSTART.md for a 5-minute guide")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        print()
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("If problems persist, see INSTALL.md for detailed instructions.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
