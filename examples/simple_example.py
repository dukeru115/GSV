#!/usr/bin/env python3
"""
Simple Example - Getting Started with GSV 2.0

This minimal example demonstrates the basic usage pattern of GSV 2.0.
It runs a simple simulation showing how to integrate GSV with an agent.
"""

import sys
import os

# Add parent directory to path so we can import gsv2 modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsv2_core import GSV2Controller, GSV2Params, MetricComputer, ModulationFunctions
import numpy as np


def main():
    """Run a simple GSV demonstration"""
    
    print("=" * 70)
    print("GSV 2.0 - Simple Example")
    print("=" * 70)
    print()
    
    # Initialize GSV with conservative parameters (more stable)
    print("1. Initializing GSV with conservative parameters...")
    params = GSV2Params.conservative()
    gsv = GSV2Controller(params)
    metrics_computer = MetricComputer()
    
    print(f"   Parameters: γE={params.gamma[1]:.3f}, kAE={params.k_AE:.3f}")
    print(f"   Stability condition: γE > kAE? {params.gamma[1] > params.k_AE}")
    print()
    
    # Simulate agent training loop
    print("2. Running simulation (100 steps)...")
    print()
    
    n_steps = 100
    
    for step in range(n_steps):
        # Simulate agent producing metrics
        # In real usage, these come from your actual agent
        td_error = np.random.randn() * 0.5  # Simulated TD error
        policy = np.random.dirichlet([1, 1, 1, 1])  # Random policy distribution
        state_id = np.random.randint(0, 20)  # Simulated state
        reward = np.random.randn() * 0.1  # Simulated reward
        
        # Compute GSV metrics
        stress = metrics_computer.compute_stress(td_error)
        coherence = metrics_computer.compute_coherence(policy)
        novelty = metrics_computer.compute_novelty(state_id)
        fitness = metrics_computer.compute_fitness(reward)
        
        # Package metrics
        metrics = {
            'rho_def': stress,
            'R': coherence,
            'novelty': novelty,
            'SIR': 0.0,  # For multi-agent scenarios
            'F': fitness
        }
        
        # Update GSV (slow timescale dynamics)
        gsv.step(metrics, dt=1.0)
        
        # Get current GSV state
        state = gsv.get_state()
        
        # Modulate agent parameters based on GSV state
        epsilon = ModulationFunctions.epsilon_exploration(state['E'])
        alpha = ModulationFunctions.alpha_learning(state['P'])
        gamma_discount = ModulationFunctions.gamma_discount(state['A'])
        
        # Print periodic updates
        if step % 20 == 0:
            print(f"   Step {step:3d}:")
            print(f"     GSV State: A={state['A']:+.3f}, E={state['E']:+.3f}, "
                  f"P={state['P']:+.3f}, S={state['S']:+.3f}")
            print(f"     Modulated Parameters: ε={epsilon:.3f}, α={alpha:.6f}, γ={gamma_discount:.3f}")
            print(f"     Metrics: stress={stress:.3f}, coherence={coherence:.3f}")
            print()
    
    # Final state
    print("3. Final Results:")
    state = gsv.get_state()
    lyapunov = gsv.compute_lyapunov()
    print(f"   Final GSV State: A={state['A']:+.3f}, E={state['E']:+.3f}, "
          f"P={state['P']:+.3f}, S={state['S']:+.3f}")
    print(f"   Lyapunov Function V(S) = {lyapunov:.6f}")
    print()
    
    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Try running: python gsv2_gridworld_demo.py")
    print("  - See gsv2_quickstart.py for more examples")
    print("  - Read README.md for full documentation")


if __name__ == "__main__":
    main()
