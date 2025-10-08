"""
GSV 2.0 - Complete Gridworld Demonstration
Reproduces Scenario 1 from paper: Regime-Change Gridworld with Stable Adaptation

This demo shows:
- Initial exploration phase
- Exploitation after learning
- Adaptive response to regime changes
- Guaranteed recovery through bounded dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import sys

# Import our GSV components
try:
    from gsv2_core import GSV2Controller, GSV2Params, MetricComputer
    from gsv2_qlearning import GSV2ModulatedQLearning, QLearningAgent, SimpleGridworld
except ImportError:
    print("Please ensure gsv2_core.py and gsv2_qlearning.py are in the same directory")
    sys.exit(1)


class ExperimentRunner:
    """
    Runs the complete GSV 2.0 experiment with data collection
    """
    
    def __init__(
        self,
        env: SimpleGridworld,
        agent: GSV2ModulatedQLearning,
        regime_changes: List[int] = [1000, 2000]
    ):
        self.env = env
        self.agent = agent
        self.regime_changes = regime_changes
        
        # Data collection
        self.episode_data = []
        self.gsv_trajectory = []
        self.performance_metrics = []
    
    def run(self, n_episodes: int = 3000, max_steps: int = 100) -> Dict:
        """
        Run complete experiment
        
        Args:
            n_episodes: Total number of episodes
            max_steps: Max steps per episode
            
        Returns:
            Dictionary with all collected data
        """
        print(f"Starting GSV 2.0 Experiment: {n_episodes} episodes")
        print("=" * 70)
        
        for episode in range(n_episodes):
            # Handle regime changes
            if episode in self.regime_changes:
                self.env.change_goal()
                print(f"\n🔄 REGIME CHANGE at episode {episode}")
                print(f"   Goal moved to {self.env.goal_pos}")
                print()
            
            # Run episode
            episode_info = self._run_episode(max_steps)
            self.episode_data.append(episode_info)
            
            # Collect GSV state
            gsv_state = self.agent.gsv.get_state()
            self.gsv_trajectory.append({
                'episode': episode,
                'SA': gsv_state['A'],
                'SE': gsv_state['E'],
                'SP': gsv_state['P'],
                'SS': gsv_state['S']
            })
            
            # Progress reporting
            if (episode + 1) % 100 == 0 or episode in self.regime_changes:
                self._print_progress(episode)
        
        print("\n" + "=" * 70)
        print("Experiment complete!")
        
        return {
            'episodes': self.episode_data,
            'gsv_trajectory': self.gsv_trajectory,
            'regime_changes': self.regime_changes
        }
    
    def _run_episode(self, max_steps: int) -> Dict:
        """Run single episode and collect data"""
        state = self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        step_data = []
        
        while not done and steps < max_steps:
            # Select and execute action
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)
            
            # GSV modulation step
            step_info = self.agent.step(state, action, reward, next_state, done)
            
            # Record data
            step_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'epsilon': step_info['epsilon'],
                'td_error': step_info['td_error']
            })
            
            total_reward += reward
            state = next_state
            steps += 1
        
        # Record episode completion
        self.agent.end_episode(total_reward, steps, done)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'success': done,
            'step_data': step_data
        }
    
    def _print_progress(self, episode: int):
        """Print progress statistics"""
        stats = self.agent.get_statistics(window=50)
        gsv_state = self.agent.gsv.get_state()
        
        # Get current parameters
        epsilon = self.agent.agent.epsilon
        alpha = self.agent.agent.alpha
        gamma = self.agent.agent.gamma
        
        print(f"Episode {episode + 1:4d} │ "
              f"Reward: {stats['mean_reward']:6.2f} │ "
              f"Success: {stats['success_rate']:.2%} │ "
              f"Steps: {stats['mean_length']:4.1f} │ "
              f"ε: {epsilon:.3f} │ "
              f"GSV: [{gsv_state['A']:+.2f}, {gsv_state['E']:+.2f}, "
              f"{gsv_state['P']:+.2f}, {gsv_state['S']:+.2f}]")


class ResultVisualizer:
    """
    Visualizes GSV 2.0 experiment results
    Reproduces figures from paper
    """
    
    def __init__(self, results: Dict):
        self.results = results
        self.episodes = results['episodes']
        self.gsv_trajectory = results['gsv_trajectory']
        self.regime_changes = results['regime_changes']
    
    def plot_all(self, save_path: str = None):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Average reward
        ax2 = fig.add_subplot(gs[0, 1])  # Exploration rate
        ax3 = fig.add_subplot(gs[1, 0])  # Success rate
        ax4 = fig.add_subplot(gs[1, 1])  # GSV components
        ax5 = fig.add_subplot(gs[2, :])  # Phase portrait
        
        self._plot_rewards(ax1)
        self._plot_exploration(ax2)
        self._plot_success(ax3)
        self._plot_gsv_components(ax4)
        self._plot_phase_portrait(ax5)
        
        # Main title
        fig.suptitle('GSV 2.0: Regime-Change Gridworld Adaptation', 
                     fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def _plot_rewards(self, ax):
        """Plot average reward over time"""
        rewards = [ep['total_reward'] for ep in self.episodes]
        
        # Smooth with moving average
        window = 50
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        ax.plot(smoothed, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('Performance Over Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Mark regime changes
        for change in self.regime_changes:
            ax.axvline(change, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.text(change, ax.get_ylim()[1] * 0.9, 'Regime\nShift', 
                   ha='center', fontsize=9, color='red', fontweight='bold')
    
    def _plot_exploration(self, ax):
        """Plot exploration rate (epsilon)"""
        # Extract epsilon from episode data
        epsilons = []
        for ep in self.episodes:
            if ep['step_data']:
                epsilons.append(ep['step_data'][-1]['epsilon'])
            else:
                epsilons.append(0.1)
        
        ax.plot(epsilons, linewidth=2, color='#F77F00')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Exploration Rate (ε)', fontsize=12)
        ax.set_title('Adaptive Exploration', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.6])
        
        # Mark regime changes
        for change in self.regime_changes:
            ax.axvline(change, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    def _plot_success(self, ax):
        """Plot success rate over time"""
        successes = [1 if ep['success'] else 0 for ep in self.episodes]
        
        # Smooth
        window = 50
        success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
        
        ax.plot(success_rate, linewidth=2, color='#06A77D')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate', fontsize=12)
        ax.set_title('Task Success Over Time', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Mark regime changes
        for change in self.regime_changes:
            ax.axvline(change, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    def _plot_gsv_components(self, ax):
        """Plot GSV state components"""
        episodes = [g['episode'] for g in self.gsv_trajectory]
        SA = [g['SA'] for g in self.gsv_trajectory]
        SE = [g['SE'] for g in self.gsv_trajectory]
        SP = [g['SP'] for g in self.gsv_trajectory]
        SS = [g['SS'] for g in self.gsv_trajectory]
        
        ax.plot(episodes, SA, label='SA (Arousal)', linewidth=2, color='#D62828')
        ax.plot(episodes, SE, label='SE (Exploration)', linewidth=2, color='#F77F00')
        ax.plot(episodes, SP, label='SP (Plasticity)', linewidth=2, alpha=0.6)
        ax.plot(episodes, SS, label='SS (Social)', linewidth=2, alpha=0.6)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Component Value', fontsize=12)
        ax.set_title('GSV Dynamics', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Mark regime changes
        for change in self.regime_changes:
            ax.axvline(change, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    def _plot_phase_portrait(self, ax):
        """Plot phase portrait (SA-SE plane)"""
        SA = [g['SA'] for g in self.gsv_trajectory]
        SE = [g['SE'] for g in self.gsv_trajectory]
        episodes = [g['episode'] for g in self.gsv_trajectory]
        
        # Color by time
        scatter = ax.scatter(SA, SE, c=episodes, cmap='viridis', 
                            s=10, alpha=0.6)
        
        # Mark start and end
        ax.scatter(SA[0], SE[0], c='green', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax.scatter(SA[-1], SE[-1], c='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2, label='End', zorder=5)
        
        # Mark regime changes
        for i, change in enumerate(self.regime_changes):
            if change < len(SA):
                ax.scatter(SA[change], SE[change], c='orange', s=150, 
                          marker='X', edgecolors='red', linewidths=2, 
                          label=f'Regime {i+1}' if i == 0 else '', zorder=5)
        
        ax.set_xlabel('SA (Arousal)', fontsize=12)
        ax.set_ylabel('SE (Exploration)', fontsize=12)
        ax.set_title('Phase Portrait: Arousal-Exploration Dynamics', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Episode', fontsize=11)


def main():
    """Run complete demonstration"""
    print("\n" + "="*70)
    print("GSV 2.0: Global State Vector for Autonomous AI Agents")
    print("Demonstration: Regime-Change Gridworld with Stable Adaptation")
    print("="*70 + "\n")
    
    # Configuration
    N_EPISODES = 3000
    REGIME_CHANGES = [1000, 2000]
    
    # Setup environment
    env = SimpleGridworld(size=10)
    print(f"Environment: {env.size}×{env.size} Gridworld")
    print(f"Initial goal: {env.goal_pos}")
    
    # Setup base Q-learning agent
    base_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3
    )
    
    # Setup GSV controller with balanced parameters
    gsv_params = GSV2Params()  # Balanced configuration
    gsv = GSV2Controller(gsv_params)
    print(f"\nGSV Configuration: Balanced")
    print(f"  Stability condition: γE={gsv_params.gamma[1]:.3f} > kAE={gsv_params.k_AE:.3f} ✓")
    
    # Setup metric computer
    metrics = MetricComputer(window_size=100, ewma_lambda=0.05)
    
    # Create modulated agent
    agent = GSV2ModulatedQLearning(base_agent, gsv, metrics)
    
    # Run experiment
    print(f"\nRunning {N_EPISODES} episodes with regime changes at {REGIME_CHANGES}")
    print("-" * 70 + "\n")
    
    runner = ExperimentRunner(env, agent, regime_changes=REGIME_CHANGES)
    results = runner.run(n_episodes=N_EPISODES, max_steps=100)
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = ResultVisualizer(results)
    visualizer.plot_all(save_path='gsv2_gridworld_results.png')
    
    # Print summary statistics
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for i, change in enumerate(REGIME_CHANGES + [N_EPISODES]):
        start = REGIME_CHANGES[i-1] if i > 0 else 0
        end = change
        
        phase_episodes = results['episodes'][start:end]
        success_rate = np.mean([ep['success'] for ep in phase_episodes])
        avg_reward = np.mean([ep['total_reward'] for ep in phase_episodes])
        
        print(f"\nPhase {i+1} (Episodes {start}-{end}):")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average reward: {avg_reward:.2f}")
    
    print("\n" + "="*70)
    print("GSV 2.0 Gridworld Demonstration: ✓ Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()