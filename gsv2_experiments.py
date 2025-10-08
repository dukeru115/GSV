"""
GSV 2.0 - Experiment Utilities
Tools for:
- Parameter comparison
- Ablation studies
- Sensitivity analysis
- Performance benchmarking
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable, Optional, Tuple
from dataclasses import dataclass
import time

try:
    from gsv2_core import GSV2Controller, GSV2Params, MetricComputer
    from gsv2_qlearning import GSV2ModulatedQLearning, QLearningAgent, SimpleGridworld
except ImportError:
    print("Please ensure gsv2_core.py and gsv2_qlearning.py are available")
    import sys
    sys.exit(1)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    gsv_params: GSV2Params
    n_episodes: int = 1000
    regime_changes: List[int] = None
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.regime_changes is None:
            self.regime_changes = []


class ParameterComparison:
    """
    Compare different parameter configurations
    """
    
    def __init__(self, env_factory: Callable, configs: List[ExperimentConfig]):
        self.env_factory = env_factory
        self.configs = configs
        self.results = {}
    
    def run_all(self) -> Dict:
        """
        Run all configurations and collect results
        
        Returns:
            Dictionary mapping config names to results
        """
        print("\n" + "=" * 70)
        print("PARAMETER COMPARISON EXPERIMENT")
        print("=" * 70 + "\n")
        
        for config in self.configs:
            print(f"Running configuration: {config.name}")
            print("-" * 70)
            
            if config.seed is not None:
                np.random.seed(config.seed)
            
            # Setup
            env = self.env_factory()
            base_agent = QLearningAgent(
                n_states=env.n_states,
                n_actions=env.n_actions
            )
            gsv = GSV2Controller(config.gsv_params)
            metrics = MetricComputer()
            agent = GSV2ModulatedQLearning(base_agent, gsv, metrics)
            
            # Run episodes
            episode_rewards = []
            episode_success = []
            
            for episode in range(config.n_episodes):
                # Handle regime changes
                if episode in config.regime_changes:
                    env.change_goal()
                
                # Run episode
                state = env.reset()
                total_reward = 0
                steps = 0
                done = False
                
                while not done and steps < 100:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    
                    total_reward += reward
                    state = next_state
                    steps += 1
                
                agent.end_episode(total_reward, steps, done)
                episode_rewards.append(total_reward)
                episode_success.append(1 if done else 0)
                
                if (episode + 1) % 100 == 0:
                    recent_success = np.mean(episode_success[-50:])
                    print(f"  Episode {episode + 1}: Success rate = {recent_success:.2%}")
            
            # Store results
            self.results[config.name] = {
                'rewards': episode_rewards,
                'success': episode_success,
                'gsv_history': gsv.history,
                'config': config
            }
            
            print(f"  Final success rate: {np.mean(episode_success[-100:]):.2%}")
            print()
        
        print("=" * 70)
        print("All configurations complete!\n")
        
        return self.results
    
    def plot_comparison(self, save_path: str = None):
        """Create comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Reward over time
        ax = axes[0, 0]
        for name, result in self.results.items():
            rewards = result['rewards']
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            ax.plot(smoothed, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success rate
        ax = axes[0, 1]
        for name, result in self.results.items():
            success = result['success']
            smoothed = np.convolve(success, np.ones(50)/50, mode='valid')
            ax.plot(smoothed, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Plot 3: GSV trajectory norms
        ax = axes[1, 0]
        for name, result in self.results.items():
            history = np.array(result['gsv_history'])
            norms = np.linalg.norm(history, axis=1)
            ax.plot(norms, label=name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('||S||')
        ax.set_title('GSV State Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final statistics
        ax = axes[1, 1]
        names = list(self.results.keys())
        final_success = [np.mean(r['success'][-100:]) for r in self.results.values()]
        
        bars = ax.bar(range(len(names)), final_success, color='#06A77D', alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Success Rate')
        ax.set_title('Final Success Rate (Last 100 Episodes)')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, final_success)):
            ax.text(i, val + 0.02, f'{val:.2%}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print summary statistics"""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70 + "\n")
        
        for name, result in self.results.items():
            rewards = result['rewards']
            success = result['success']
            
            print(f"{name}:")
            print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Final success rate: {np.mean(success[-100:]):.2%}")
            print(f"  Adaptation speed (episodes to 50% success):")
            
            # Find when success rate first reaches 50%
            smoothed_success = np.convolve(success, np.ones(50)/50, mode='valid')
            idx_50 = np.where(smoothed_success >= 0.5)[0]
            if len(idx_50) > 0:
                print(f"    {idx_50[0]} episodes")
            else:
                print(f"    Never reached 50%")
            
            print()


class AblationStudy:
    """
    Ablation study: test effect of removing components
    """
    
    def __init__(self, env_factory: Callable):
        self.env_factory = env_factory
        self.results = {}
    
    def run(self, n_episodes: int = 500) -> Dict:
        """
        Run ablation study
        
        Tests:
        1. Full GSV (baseline)
        2. No cross-coupling (kAE = kPS = 0)
        3. No stochastic terms (σ = 0)
        4. No cubic damping (λ = 0, with hard bounds)
        5. Fixed hyperparameters (no GSV)
        """
        print("\n" + "=" * 70)
        print("ABLATION STUDY")
        print("=" * 70 + "\n")
        
        # Configuration 1: Full GSV
        configs = [
            ("Full GSV", self._create_full_config()),
            ("No Cross-Coupling", self._create_no_coupling_config()),
            ("No Stochastic", self._create_no_stochastic_config()),
            ("Fixed Hyperparams", None)  # Special case
        ]
        
        for name, params in configs:
            print(f"Testing: {name}")
            print("-" * 70)
            
            if name == "Fixed Hyperparams":
                result = self._run_fixed_hyperparams(n_episodes)
            else:
                result = self._run_with_params(params, n_episodes)
            
            self.results[name] = result
            
            success_rate = np.mean(result['success'][-100:])
            print(f"  Final success rate: {success_rate:.2%}\n")
        
        print("=" * 70)
        print("Ablation study complete!\n")
        
        return self.results
    
    def _create_full_config(self) -> GSV2Params:
        """Full GSV configuration"""
        return GSV2Params()
    
    def _create_no_coupling_config(self) -> GSV2Params:
        """No cross-coupling"""
        params = GSV2Params()
        params.k_AE = 0.0
        params.k_PS = 0.0
        return params
    
    def _create_no_stochastic_config(self) -> GSV2Params:
        """No stochastic terms"""
        params = GSV2Params()
        params.sigma = np.zeros(4)
        return params
    
    def _run_with_params(self, params: GSV2Params, n_episodes: int) -> Dict:
        """Run with given parameters"""
        env = self.env_factory()
        base_agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
        gsv = GSV2Controller(params)
        metrics = MetricComputer()
        agent = GSV2ModulatedQLearning(base_agent, gsv, metrics)
        
        rewards = []
        success = []
        
        for episode in range(n_episodes):
            if episode == n_episodes // 2:
                env.change_goal()  # Regime change
            
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                steps += 1
            
            agent.end_episode(total_reward, steps, done)
            rewards.append(total_reward)
            success.append(1 if done else 0)
        
        return {'rewards': rewards, 'success': success}
    
    def _run_fixed_hyperparams(self, n_episodes: int) -> Dict:
        """Run with fixed hyperparameters (no GSV)"""
        env = self.env_factory()
        agent = QLearningAgent(
            n_states=env.n_states,
            n_actions=env.n_actions,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.2  # Fixed exploration
        )
        
        rewards = []
        success = []
        
        for episode in range(n_episodes):
            if episode == n_episodes // 2:
                env.change_goal()  # Regime change
            
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 100:
                action = agent.get_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                steps += 1
            
            rewards.append(total_reward)
            success.append(1 if done else 0)
        
        return {'rewards': rewards, 'success': success}
    
    def plot_results(self, save_path: str = None):
        """Plot ablation results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Success rate
        ax = axes[0]
        for name, result in self.results.items():
            success = result['success']
            smoothed = np.convolve(success, np.ones(50)/50, mode='valid')
            ax.plot(smoothed, label=name, linewidth=2, alpha=0.8)
        
        ax.axvline(250, color='red', linestyle='--', alpha=0.5, label='Regime Change')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Ablation Study: Success Rate', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Plot 2: Final comparison
        ax = axes[1]
        names = list(self.results.keys())
        final_success = [np.mean(r['success'][-100:]) for r in self.results.values()]
        
        colors = ['#06A77D', '#2E86AB', '#F77F00', '#D62828']
        bars = ax.bar(range(len(names)), final_success, color=colors, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Final Performance', fontsize=13, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, final_success)):
            ax.text(i, val + 0.02, f'{val:.2%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class SimpleSensitivityAnalysis:
    """
    Simple sensitivity analysis for key parameters
    """
    
    def __init__(self, env_factory: Callable):
        self.env_factory = env_factory
    
    def analyze_parameter(
        self,
        param_name: str,
        param_path: List[str],
        values: np.ndarray,
        n_episodes: int = 300,
        n_runs: int = 3
    ) -> Dict:
        """
        Analyze sensitivity to a single parameter
        
        Args:
            param_name: Name of parameter for display
            param_path: Path to parameter (e.g., ['gamma', 1] for gamma[1])
            values: Array of values to test
            n_episodes: Episodes per run
            n_runs: Runs per value (for averaging)
            
        Returns:
            Dictionary with results
        """
        print(f"\nAnalyzing sensitivity to {param_name}")
        print(f"Testing {len(values)} values × {n_runs} runs = {len(values) * n_runs} experiments")
        print("-" * 70)
        
        results = {
            'values': values,
            'mean_success': [],
            'std_success': []
        }
        
        for value in values:
            run_success = []
            
            for run in range(n_runs):
                # Create params with modified value
                params = GSV2Params()
                obj = params
                for key in param_path[:-1]:
                    obj = getattr(obj, key)
                
                if isinstance(obj, np.ndarray):
                    obj[param_path[-1]] = value
                else:
                    setattr(obj, param_path[-1], value)
                
                # Run experiment
                env = self.env_factory()
                base_agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
                gsv = GSV2Controller(params)
                metrics = MetricComputer()
                agent = GSV2ModulatedQLearning(base_agent, gsv, metrics)
                
                success = []
                for episode in range(n_episodes):
                    state = env.reset()
                    done = False
                    steps = 0
                    
                    while not done and steps < 100:
                        action = agent.act(state)
                        next_state, reward, done = env.step(action)
                        agent.step(state, action, reward, next_state, done)
                        state = next_state
                        steps += 1
                    
                    success.append(1 if done else 0)
                
                final_success = np.mean(success[-50:])
                run_success.append(final_success)
            
            results['mean_success'].append(np.mean(run_success))
            results['std_success'].append(np.std(run_success))
            
            print(f"  {param_name} = {value:.4f}: "
                  f"Success = {results['mean_success'][-1]:.2%} ± {results['std_success'][-1]:.2%}")
        
        return results
    
    def plot_sensitivity(self, param_name: str, results: Dict, save_path: str = None):
        """Plot sensitivity analysis results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        values = results['values']
        mean = results['mean_success']
        std = results['std_success']
        
        ax.plot(values, mean, 'o-', linewidth=2, markersize=8, color='#06A77D')
        ax.fill_between(values, 
                        np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std),
                        alpha=0.3, color='#06A77D')
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('Final Success Rate', fontsize=12)
        ax.set_title(f'Parameter Sensitivity: {param_name}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


# Demo usage
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("GSV 2.0 EXPERIMENT UTILITIES DEMO")
    print("=" * 70)
    
    # Define environment factory
    def create_env():
        return SimpleGridworld(size=5)
    
    # Demo 1: Parameter comparison
    print("\n1. PARAMETER COMPARISON")
    configs = [
        ExperimentConfig("Balanced", GSV2Params(), n_episodes=300, regime_changes=[150]),
        ExperimentConfig("Conservative", GSV2Params.conservative(), n_episodes=300, regime_changes=[150]),
        ExperimentConfig("Aggressive", GSV2Params.aggressive(), n_episodes=300, regime_changes=[150]),
    ]
    
    comparison = ParameterComparison(create_env, configs)
    comparison.run_all()
    comparison.print_summary()
    comparison.plot_comparison(save_path='gsv2_comparison.png')
    
    # Demo 2: Ablation study
    print("\n2. ABLATION STUDY")
    ablation = AblationStudy(create_env)
    ablation.run(n_episodes=300)
    ablation.plot_results(save_path='gsv2_ablation.png')
    
    print("\n" + "=" * 70)
    print("Experiment Utilities Demo: ✓ Complete")
    print("=" * 70 + "\n")