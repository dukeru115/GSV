"""
GSV 2.0 - Q-Learning Agent Integration
Demonstrates closed-loop adaptation with strategic control
"""

import numpy as np
from typing import Tuple, Optional, Dict
from collections import defaultdict


class QLearningAgent:
    """
    Standard Q-Learning agent with modifiable hyperparameters
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
        
        # Hyperparameters (will be modulated by GSV)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Track last TD error for metrics
        self.last_td_error = 0.0
        
    def get_action(self, state: int) -> int:
        """
        ε-greedy action selection
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def get_policy_probs(self, state: int) -> np.ndarray:
        """
        Get current policy probabilities for coherence metric
        
        Returns ε-greedy probabilities
        """
        probs = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1 - self.epsilon)
        return probs
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Q-learning update
        
        Returns:
            TD error for metrics
        """
        # Compute TD target
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD error
        td_error = target - self.Q[state, action]
        
        # Q-value update
        self.Q[state, action] += self.alpha * td_error
        
        self.last_td_error = td_error
        return td_error
    
    def set_epsilon(self, epsilon: float):
        """Modulate exploration rate"""
        self.epsilon = np.clip(epsilon, 0.0, 1.0)
    
    def set_alpha(self, alpha: float):
        """Modulate learning rate"""
        self.alpha = np.clip(alpha, 0.0, 1.0)
    
    def set_gamma(self, gamma: float):
        """Modulate discount factor"""
        self.gamma = np.clip(gamma, 0.0, 1.0)


class GSV2ModulatedQLearning:
    """
    Q-Learning agent with GSV 2.0 strategic control
    Implements closed-loop dynamics: Agent -> Metrics -> GSV -> Parameters -> Agent
    """
    
    def __init__(
        self,
        base_agent: QLearningAgent,
        gsv_controller,
        metric_computer,
        modulation_config: Optional[Dict] = None
    ):
        self.agent = base_agent
        self.gsv = gsv_controller
        self.metrics = metric_computer
        
        # Modulation configuration
        default_config = {
            'epsilon_min': 0.05,
            'epsilon_max': 0.5,
            'alpha_base': 0.1,
            'alpha_k': 0.3,
            'gamma_base': 0.9,
            'gamma_k': 0.05
        }
        self.config = modulation_config or default_config
        
        # Track episode statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
    
    def step(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        dt: float = 1.0
    ) -> Dict:
        """
        Single step with GSV modulation
        
        Returns:
            Dictionary with step information and GSV state
        """
        # 1. Agent learns (produces TD error)
        td_error = self.agent.update(state, action, reward, next_state, done)
        
        # 2. Compute metrics for GSV
        stress = self.metrics.compute_stress(td_error)
        coherence = self.metrics.compute_coherence(self.agent.get_policy_probs(state))
        novelty = self.metrics.compute_novelty(state)
        fitness = self.metrics.compute_fitness(reward)
        
        gsv_metrics = {
            'rho_def': stress,
            'R': coherence,
            'novelty': novelty,
            'SIR': 0.0,  # Not used in single-agent
            'F': fitness
        }
        
        # 3. Update GSV (slow dynamics)
        self.gsv.step(gsv_metrics, dt=dt)
        
        # 4. Modulate agent parameters based on new GSV state
        gsv_state = self.gsv.get_state()
        
        from gsv2_core import ModulationFunctions as MF
        
        new_epsilon = MF.epsilon_exploration(
            gsv_state['E'],
            self.config['epsilon_min'],
            self.config['epsilon_max']
        )
        
        new_alpha = MF.alpha_learning(
            gsv_state['P'],
            self.config['alpha_base'],
            self.config['alpha_k']
        )
        
        new_gamma = MF.gamma_discount(
            gsv_state['A'],
            self.config['gamma_base'],
            self.config['gamma_k']
        )
        
        self.agent.set_epsilon(new_epsilon)
        self.agent.set_alpha(new_alpha)
        self.agent.set_gamma(new_gamma)
        
        # Return step info
        return {
            'td_error': td_error,
            'metrics': gsv_metrics,
            'gsv_state': gsv_state,
            'epsilon': new_epsilon,
            'alpha': new_alpha,
            'gamma': new_gamma
        }
    
    def act(self, state: int) -> int:
        """Select action using current policy"""
        return self.agent.get_action(state)
    
    def end_episode(self, total_reward: float, steps: int, success: bool):
        """Record episode statistics"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.episode_success.append(1.0 if success else 0.0)
    
    def get_statistics(self, window: int = 100) -> Dict:
        """Get recent performance statistics"""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent = slice(-window, None)
        
        return {
            'mean_reward': np.mean(self.episode_rewards[recent]),
            'mean_length': np.mean(self.episode_lengths[recent]),
            'success_rate': np.mean(self.episode_success[recent]),
            'total_episodes': len(self.episode_rewards)
        }


class SimpleGridworld:
    """
    Simple gridworld environment for testing GSV 2.0
    Features regime changes (goal location shifts)
    """
    
    def __init__(self, size: int = 10):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # up, right, down, left
        
        self.agent_pos = [0, 0]
        self.goal_pos = [9, 9]
        self.initial_goal = [9, 9]
        
    def reset(self) -> int:
        """Reset environment"""
        self.agent_pos = [0, 0]
        return self._get_state()
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Execute action
        
        Args:
            action: 0=up, 1=right, 2=down, 3=left
            
        Returns:
            (next_state, reward, done)
        """
        # Action effects
        moves = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        new_pos = [
            self.agent_pos[0] + moves[action][0],
            self.agent_pos[1] + moves[action][1]
        ]
        
        # Boundary check
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.agent_pos = new_pos
        
        # Check if goal reached
        if self.agent_pos == self.goal_pos:
            return self._get_state(), 10.0, True
        else:
            return self._get_state(), -0.1, False
    
    def change_goal(self, new_goal: Optional[list] = None):
        """
        Change goal position (regime shift)
        
        Args:
            new_goal: New goal position [row, col], or None for opposite corner
        """
        if new_goal is not None:
            self.goal_pos = new_goal
        else:
            # Toggle between corners
            if self.goal_pos == [9, 9]:
                self.goal_pos = [0, 9]
            elif self.goal_pos == [0, 9]:
                self.goal_pos = [9, 0]
            elif self.goal_pos == [9, 0]:
                self.goal_pos = [0, 0]
            else:
                self.goal_pos = [9, 9]
    
    def _get_state(self) -> int:
        """Convert position to state ID"""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def get_position(self, state: int) -> Tuple[int, int]:
        """Convert state ID to position"""
        return (state // self.size, state % self.size)


# Demo usage
if __name__ == "__main__":
    from gsv2_core import GSV2Controller, GSV2Params, MetricComputer
    
    print("GSV 2.0 Q-Learning Integration Demo")
    print("=" * 60)
    
    # Setup
    env = SimpleGridworld(size=10)
    base_agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3
    )
    
    gsv = GSV2Controller(GSV2Params())
    metrics = MetricComputer()
    
    agent = GSV2ModulatedQLearning(base_agent, gsv, metrics)
    
    print("\nRunning 100 episodes with regime change at episode 50...")
    
    for episode in range(100):
        # Regime change
        if episode == 50:
            env.change_goal()
            print("\n⚠️  REGIME CHANGE! Goal moved to opposite corner\n")
        
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            
            step_info = agent.step(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
        
        agent.end_episode(total_reward, steps, done)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            stats = agent.get_statistics(window=10)
            gsv_state = gsv.get_state()
            
            print(f"Episode {episode + 1:3d} | "
                  f"Reward: {stats['mean_reward']:6.2f} | "
                  f"Success: {stats['success_rate']:.2f} | "
                  f"ε: {step_info['epsilon']:.3f} | "
                  f"SA: {gsv_state['A']:+.2f} | "
                  f"SE: {gsv_state['E']:+.2f}")
    
    print("\n" + "=" * 60)
    print("GSV 2.0 Q-Learning Integration: ✓ Complete")