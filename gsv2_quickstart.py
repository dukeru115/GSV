"""
GSV 2.0 - Quick Start Guide
Practical examples for getting started with GSV 2.0

Includes:
1. Minimal example
2. Custom agent integration
3. Metric computation for different domains
4. Parameter tuning guidelines
5. Common patterns and best practices
"""

import numpy as np
from typing import Dict, Tuple


# ============================================================================
# EXAMPLE 1: Minimal Working Example (10 lines)
# ============================================================================

def minimal_example():
    """Absolute minimal GSV 2.0 usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Minimal Working Example")
    print("="*70 + "\n")
    
    from gsv2_core import GSV2Controller, ModulationFunctions
    
    # 1. Create controller
    gsv = GSV2Controller()
    
    # 2. Agent loop (pseudo-code structure)
    for step in range(100):
        # Your agent produces metrics
        metrics = {'rho_def': 0.5, 'R': 0.7, 'novelty': 0.2, 'SIR': 0.0, 'F': 1.0}
        
        # 3. Update GSV
        gsv.step(metrics, dt=1.0)
        
        # 4. Modulate parameters
        state = gsv.get_state()
        epsilon = ModulationFunctions.epsilon_exploration(state['E'])
        
        # Use epsilon in your agent...
        if step % 20 == 0:
            print(f"Step {step}: epsilon={epsilon:.3f}, SA={state['A']:+.2f}, SE={state['E']:+.2f}")
    
    print("\n✓ Minimal example complete\n")


# ============================================================================
# EXAMPLE 2: Custom Agent Integration Pattern
# ============================================================================

class CustomDQNAgent:
    """Example template for integrating GSV with your custom agent"""
    
    def __init__(self):
        from gsv2_core import GSV2Controller, MetricComputer
        
        # Your agent's components
        self.q_network = None  # Your neural network
        self.optimizer = None  # Your optimizer
        self.epsilon = 0.1
        self.learning_rate = 0.001
        
        # GSV components
        self.gsv = GSV2Controller()
        self.metrics = MetricComputer()
    
    def train_step(self, batch: Dict) -> Dict:
        """
        Single training step with GSV modulation
        
        Args:
            batch: Training batch from replay buffer
            
        Returns:
            Dictionary with training info
        """
        # 1. Your agent's training logic
        loss = self._compute_loss(batch)
        td_errors = self._get_td_errors(batch)
        
        # 2. Compute GSV metrics
        stress = self.metrics.compute_stress(np.mean(np.abs(td_errors)))
        coherence = self.metrics.compute_coherence(self._get_policy_probs(batch))
        novelty = self.metrics.compute_novelty(batch['states'][0])
        fitness = self.metrics.compute_fitness(np.mean(batch['rewards']))
        
        gsv_metrics = {
            'rho_def': stress,
            'R': coherence,
            'novelty': novelty,
            'SIR': 0.0,
            'F': fitness
        }
        
        # 3. Update GSV (slow timescale)
        self.gsv.step(gsv_metrics, dt=1.0)
        
        # 4. Modulate agent parameters
        from gsv2_core import ModulationFunctions as MF
        gsv_state = self.gsv.get_state()
        
        self.epsilon = MF.epsilon_exploration(gsv_state['E'])
        new_lr = MF.alpha_learning(gsv_state['P'])
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return {
            'loss': loss,
            'epsilon': self.epsilon,
            'learning_rate': new_lr,
            'gsv_state': gsv_state
        }
    
    def _compute_loss(self, batch):
        """Your loss computation"""
        return np.random.rand()  # Placeholder
    
    def _get_td_errors(self, batch):
        """Your TD error computation"""
        return np.random.randn(32)  # Placeholder
    
    def _get_policy_probs(self, batch):
        """Your policy probabilities"""
        return np.random.dirichlet([1, 1, 1, 1])  # Placeholder


def custom_agent_example():
    """Demonstrate custom agent integration"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Agent Integration")
    print("="*70 + "\n")
    
    agent = CustomDQNAgent()
    
    print("Training for 10 steps...")
    for step in range(10):
        # Simulate batch
        batch = {
            'states': [np.random.randint(0, 100)],
            'actions': [0],
            'rewards': [np.random.randn()],
            'next_states': [np.random.randint(0, 100)]
        }
        
        info = agent.train_step(batch)
        
        if step % 3 == 0:
            print(f"Step {step}: ε={info['epsilon']:.3f}, "
                  f"lr={info['learning_rate']:.6f}, "
                  f"SE={info['gsv_state']['E']:+.2f}")
    
    print("\n✓ Custom agent integration complete\n")


# ============================================================================
# EXAMPLE 3: Domain-Specific Metrics
# ============================================================================

class RLMetrics:
    """Metrics for Reinforcement Learning agents"""
    
    @staticmethod
    def compute_from_trajectory(states, actions, rewards, td_errors):
        """
        Compute GSV metrics from RL trajectory
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            td_errors: List of TD errors
            
        Returns:
            Dictionary of GSV metrics
        """
        # Stress: from TD errors
        rho_def = np.clip(np.mean(np.abs(td_errors)) / 10.0, 0, 1)
        
        # Coherence: from policy entropy (approximate from action diversity)
        action_counts = np.bincount(actions)
        entropy = -np.sum((action_counts / len(actions)) * 
                         np.log(action_counts / len(actions) + 1e-10))
        max_entropy = np.log(len(np.unique(actions)))
        R = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)
        
        # Novelty: from state visitation
        unique_states = len(np.unique(states))
        novelty = unique_states / len(states)
        
        # Fitness: from rewards
        F = np.mean(rewards)
        
        return {
            'rho_def': rho_def,
            'R': R,
            'novelty': novelty,
            'SIR': 0.0,
            'F': F
        }


class LLMMetrics:
    """Metrics for Large Language Models"""
    
    @staticmethod
    def compute_from_generation(tokens, probs, perplexity):
        """
        Compute GSV metrics from LLM generation
        
        Args:
            tokens: Generated tokens
            probs: Token probabilities
            perplexity: Current perplexity
            
        Returns:
            Dictionary of GSV metrics
        """
        # Stress: from perplexity
        rho_def = np.clip(np.log(perplexity) / 10.0, 0, 1)
        
        # Coherence: from token confidence
        R = np.mean([max(p) for p in probs])
        
        # Novelty: from rare tokens
        novelty = np.mean([1 if max(p) < 0.1 else 0 for p in probs])
        
        # Social: from dialog metrics (if applicable)
        SIR = 0.0  # Compute from user engagement, etc.
        
        return {
            'rho_def': rho_def,
            'R': R,
            'novelty': novelty,
            'SIR': SIR,
            'F': -perplexity
        }


def domain_metrics_example():
    """Demonstrate domain-specific metrics"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Domain-Specific Metrics")
    print("="*70 + "\n")
    
    # RL example
    print("RL Metrics:")
    states = [0, 1, 2, 1, 3, 4, 2]
    actions = [0, 1, 0, 1, 0, 1, 0]
    rewards = [1.0, -0.5, 2.0, 0.0, 1.5, -1.0, 0.5]
    td_errors = [0.5, -0.3, 1.2, -0.1, 0.8, -0.6, 0.2]
    
    rl_metrics = RLMetrics.compute_from_trajectory(states, actions, rewards, td_errors)
    print(f"  Stress: {rl_metrics['rho_def']:.3f}")
    print(f"  Coherence: {rl_metrics['R']:.3f}")
    print(f"  Novelty: {rl_metrics['novelty']:.3f}")
    print(f"  Fitness: {rl_metrics['F']:.3f}")
    
    # LLM example
    print("\nLLM Metrics:")
    tokens = ["the", "cat", "sat", "on", "mat"]
    probs = [[0.8, 0.1, 0.05, 0.05], [0.6, 0.2, 0.1, 0.1],
             [0.7, 0.15, 0.1, 0.05], [0.9, 0.05, 0.03, 0.02],
             [0.5, 0.3, 0.1, 0.1]]
    perplexity = 12.5
    
    llm_metrics = LLMMetrics.compute_from_generation(tokens, probs, perplexity)
    print(f"  Stress: {llm_metrics['rho_def']:.3f}")
    print(f"  Coherence: {llm_metrics['R']:.3f}")
    print(f"  Novelty: {llm_metrics['novelty']:.3f}")
    
    print("\n✓ Domain metrics examples complete\n")


# ============================================================================
# EXAMPLE 4: Parameter Tuning Guidelines
# ============================================================================

def parameter_tuning_guide():
    """Guidelines for tuning GSV parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Parameter Tuning Guidelines")
    print("="*70 + "\n")
    
    print("📋 PARAMETER TUNING CHECKLIST:\n")
    
    print("1. START WITH PRESETS:")
    print("   from gsv2_core import GSV2Params")
    print("   params = GSV2Params()              # Balanced (recommended)")
    print("   params = GSV2Params.conservative() # Maximum stability")
    print("   params = GSV2Params.aggressive()   # Fast adaptation")
    print()
    
    print("2. STABILITY CONDITION (CRITICAL):")
    print("   ✓ ALWAYS ensure: γE > kAE")
    print("   - Default: γE=0.01, kAE=0.1 ✓")
    print("   - If increasing kAE, increase γE proportionally")
    print()
    
    print("3. TIMESCALE SEPARATION:")
    print("   - GSV should evolve ~100x slower than agent")
    print("   - Typical: γi ∈ [0.005, 0.02] for dt=1s")
    print("   - Adjust based on your episode length")
    print()
    
    print("4. SENSITIVITY (α) TUNING:")
    print("   - Higher α → faster response to metrics")
    print("   - Start with α ∈ [0.05, 0.1]")
    print("   - Increase if adaptation too slow")
    print("   - Decrease if too reactive/noisy")
    print()
    
    print("5. DAMPING (λ) TUNING:")
    print("   - Higher λ → tighter bounds on S")
    print("   - Start with λ ∈ [0.002, 0.005]")
    print("   - Increase if states grow too large")
    print("   - Check with: gsv.compute_lyapunov()")
    print()
    
    print("6. NOISE (σ) TUNING:")
    print("   - Higher σ → more exploration, less trapping")
    print("   - Start with σ ∈ [0.02, 0.03]")
    print("   - Increase if stuck in local minima")
    print("   - Decrease if too erratic")
    print()
    
    print("7. DIAGNOSTIC TOOLS:")
    print("   from gsv2_analysis import StabilityAnalyzer, DiagnosticMonitor")
    print("   analyzer = StabilityAnalyzer(gsv)")
    print("   monitor = DiagnosticMonitor(gsv)")
    print("   monitor.check_health()  # Real-time monitoring")
    print()
    
    print("8. VALIDATION:")
    print("   - Run for ~1000 episodes")
    print("   - Check: ||S|| < 3.0 (bounded)")
    print("   - Check: eigenvalues < 0 (stable)")
    print("   - Check: adapts to regime changes")
    print()


# ============================================================================
# EXAMPLE 5: Common Patterns
# ============================================================================

def common_patterns():
    """Common usage patterns"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Common Patterns & Best Practices")
    print("="*70 + "\n")
    
    print("PATTERN 1: Episode-Level Integration")
    print("-" * 70)
    print("""
    for episode in range(n_episodes):
        state = env.reset()
        episode_metrics = []
        
        while not done:
            # Fast loop: agent acts/learns
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            # Collect metrics at each step
            episode_metrics.append({
                'td_error': agent.last_td_error,
                'reward': reward
            })
            
            state = next_state
        
        # Slow loop: update GSV once per episode
        aggregated_metrics = aggregate(episode_metrics)
        gsv.step(aggregated_metrics, dt=1.0)
        
        # Modulate for next episode
        update_agent_params(gsv.get_state())
    """)
    
    print("\nPATTERN 2: Batch-Level Integration (for Neural Networks)")
    print("-" * 70)
    print("""
    for batch_idx in range(n_batches):
        # Training step
        loss, td_errors = model.train_step(batch)
        
        # Update GSV every N batches
        if batch_idx % update_frequency == 0:
            metrics = compute_metrics(loss, td_errors)
            gsv.step(metrics, dt=update_frequency)
            
            # Adjust learning rate
            state = gsv.get_state()
            new_lr = ModulationFunctions.alpha_learning(state['P'])
            optimizer.lr = new_lr
    """)
    
    print("\nPATTERN 3: Multi-Agent Coordination")
    print("-" * 70)
    print("""
    # Each agent has its own GSV
    agents = [create_agent_with_gsv() for _ in range(n_agents)]
    
    for episode in range(n_episodes):
        # Collect individual metrics
        for agent in agents:
            agent.act_and_learn()
        
        # Compute social metrics
        coordination_success = evaluate_coordination(agents)
        
        # Update each agent's GSV
        for agent in agents:
            agent.gsv.step({
                ...,
                'SIR': coordination_success  # Social component
            })
    """)
    
    print("\nBEST PRACTICES:")
    print("-" * 70)
    print("✓ Always validate stability conditions before long runs")
    print("✓ Monitor Lyapunov function periodically")
    print("✓ Use EWMA for metric smoothing (window ~50-100)")
    print("✓ Log GSV state alongside agent metrics")
    print("✓ Test parameter changes on short runs first")
    print("✓ Use diagnostic tools for debugging")
    print("✓ Start with balanced params, tune conservatively")
    print()


# ============================================================================
# EXAMPLE 6: Troubleshooting Guide
# ============================================================================

def troubleshooting():
    """Common issues and solutions"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Troubleshooting Guide")
    print("="*70 + "\n")
    
    issues = [
        {
            'symptom': "GSV state diverges (||S|| > 5)",
            'causes': [
                "- α too high (too sensitive to metrics)",
                "- λ too small (insufficient damping)",
                "- Metrics not normalized to [0,1]"
            ],
            'solutions': [
                "→ Decrease α by 50%",
                "→ Increase λ by 2x",
                "→ Check metric computation"
            ]
        },
        {
            'symptom': "No adaptation to regime changes",
            'causes': [
                "- γ too high (decay too fast)",
                "- α too low (not sensitive enough)",
                "- σ = 0 (no stochastic exploration)"
            ],
            'solutions': [
                "→ Decrease γ by 30%",
                "→ Increase α by 50%",
                "→ Add noise: σ ∈ [0.02, 0.05]"
            ]
        },
        {
            'symptom': "System unstable (oscillations)",
            'causes': [
                "- Violated γE > kAE condition",
                "- dt too large",
                "- Metrics too noisy"
            ],
            'solutions': [
                "→ Ensure γE > kAE",
                "→ Reduce dt (try 0.5 or 0.1)",
                "→ Increase EWMA smoothing"
            ]
        },
        {
            'symptom': "Agent performance worse than baseline",
            'causes': [
                "- Wrong metric computation",
                "- Modulation functions poorly tuned",
                "- GSV timescale too fast"
            ],
            'solutions': [
                "→ Verify metrics are in [0,1]",
                "→ Check modulation output ranges",
                "→ Increase γ (slower adaptation)"
            ]
        }
    ]
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. SYMPTOM: {issue['symptom']}")
        print("   Possible Causes:")
        for cause in issue['causes']:
            print(f"     {cause}")
        print("   Solutions:")
        for solution in issue['solutions']:
            print(f"     {solution}")
        print()


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run all quick start examples"""
    print("\n" + "🚀 " + "="*67)
    print("GSV 2.0 QUICK START GUIDE")
    print("="*70)
    
    # Run examples
    minimal_example()
    custom_agent_example()
    domain_metrics_example()
    parameter_tuning_guide()
    common_patterns()
    troubleshooting()
    
    # Final tips
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print()
    print("1. Run full demonstration:")
    print("   python gsv2_gridworld_demo.py")
    print()
    print("2. Run tests:")
    print("   python gsv2_tests.py")
    print()
    print("3. Try different scenarios:")
    print("   python gsv2_stress_demo.py")
    print()
    print("4. Experiment with parameters:")
    print("   python gsv2_experiments.py")
    print()
    print("5. Read the paper:")
    print("   Urmanov et al. (2025) - Global State Vector 2.0")
    print()
    print("="*70)
    print("✓ Quick Start Guide Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()