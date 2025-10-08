# GSV 2.0 Examples

This directory contains example scripts demonstrating how to use GSV 2.0.

## Available Examples

### 1. Simple Example (`simple_example.py`)

A minimal demonstration of GSV 2.0 basics:
- Initializing GSV controller
- Computing metrics from agent data
- Updating GSV state
- Modulating agent parameters

**Run it:**
```bash
python examples/simple_example.py
```

**Expected runtime:** < 1 second

## More Examples

For additional examples, see the main directory:

- **`gsv2_quickstart.py`**: 6 comprehensive examples with detailed explanations
- **`gsv2_gridworld_demo.py`**: Full demonstration with 3000 episodes
- **`gsv2_stress_demo.py`**: Stress response scenario
- **`gsv2_experiments.py`**: Research tools for parameter studies

## Creating Your Own Examples

Template for a new example:

```python
#!/usr/bin/env python3
"""Your example description"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsv2_core import GSV2Controller, GSV2Params, MetricComputer, ModulationFunctions

def main():
    # Initialize
    gsv = GSV2Controller(GSV2Params.conservative())
    metrics = MetricComputer()
    
    # Your code here
    for step in range(100):
        # Compute metrics from your agent
        stress = metrics.compute_stress(td_error)
        coherence = metrics.compute_coherence(policy)
        
        # Update GSV
        gsv.step({'rho_def': stress, 'R': coherence, ...})
        
        # Modulate parameters
        state = gsv.get_state()
        epsilon = ModulationFunctions.epsilon_exploration(state['E'])
        
        # Use epsilon in your agent...

if __name__ == "__main__":
    main()
```

## Contributing Examples

Have a useful example? Please contribute!

1. Create your example in this directory
2. Add documentation explaining what it demonstrates
3. Update this README
4. Submit a pull request

See `CONTRIBUTING.md` in the root directory for guidelines.
