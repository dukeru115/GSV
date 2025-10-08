"""
Global State Vector 2.0 - Core Implementation
Multi-Scale Control for Autonomous AI Agents

Based on: Urmanov, T., Gadeev, K., & Iusupov, B. (2025)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque


@dataclass
class GSV2Params:
    """Parameters for GSV 2.0 system"""
    # Sensitivity to driving signals (1/s)
    alpha: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.05, 0.03, 0.06]))
    
    # Linear decay rates (1/s)
    gamma: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.008, 0.01]))
    
    # Nonlinear damping (1/(s·S²))
    lambda_: np.ndarray = field(default_factory=lambda: np.array([0.003, 0.003, 0.002, 0.003]))
    
    # Noise intensity (S/√s)
    sigma: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.02, 0.02, 0.025]))
    
    # Cross-coupling coefficients (1/s)
    k_AE: float = 0.1  # Arousal -> Exploration
    k_PS: float = 0.05  # Social -> Plasticity
    
    # Metric-specific sensitivities
    k_A: float = 0.5   # External stimulus sensitivity
    k_P: float = 0.5   # Fitness sensitivity for plasticity
    
    # Target values
    R_target: float = 0.8  # Target coherence
    F_target: float = 0.0  # Target fitness (will be set adaptively)
    
    def __post_init__(self):
        """Validate stability conditions"""
        # Check Theorem 1: γE > kAE for stability
        if self.gamma[1] <= self.k_AE:
            print(f"WARNING: Stability condition violated! γE={self.gamma[1]} <= kAE={self.k_AE}")
            print("System may become unstable. Consider increasing gamma[1] or decreasing k_AE")
    
    @classmethod
    def conservative(cls):
        """Maximum stability configuration"""
        return cls(
            alpha=np.array([0.05, 0.03, 0.02, 0.04]),
            gamma=np.array([0.015, 0.015, 0.01, 0.01]),
            lambda_=np.array([0.005, 0.005, 0.004, 0.004]),
            sigma=np.array([0.015, 0.015, 0.015, 0.02]),
            k_AE=0.05,
            k_PS=0.03
        )
    
    @classmethod
    def aggressive(cls):
        """Fast adaptation configuration"""
        return cls(
            alpha=np.array([0.1, 0.08, 0.05, 0.08]),
            gamma=np.array([0.008, 0.008, 0.006, 0.008]),
            lambda_=np.array([0.002, 0.002, 0.0015, 0.002]),
            sigma=np.array([0.03, 0.03, 0.025, 0.03]),
            k_AE=0.15,
            k_PS=0.08
        )


class GSV2Controller:
    """
    Global State Vector 2.0 Controller
    Implements stochastic differential equations for multi-scale control
    """
    
    def __init__(self, params: Optional[GSV2Params] = None):
        self.params = params or GSV2Params()
        
        # State vector: [SA (Arousal), SE (Exploration), SP (Plasticity), SS (Social)]
        self.state = np.zeros(4)
        
        # History tracking
        self.history: List[np.ndarray] = []
        self.metrics_history: List[Dict] = []
        self.time_history: List[float] = []
        self.current_time = 0.0
    
    def step(self, metrics: Dict, dt: float = 1.0, ext_stim: float = 0.0) -> np.ndarray:
        """
        Single integration step using Euler-Maruyama method
        
        Args:
            metrics: Dictionary containing:
                - 'rho_def': Stress metric (0-1)
                - 'R': Coherence metric (0-1)
                - 'novelty': Novelty rate (0-1)
                - 'SIR': Social interaction rate (0-1)
                - 'F': Fitness/reward (any scale)
            dt: Time step (seconds)
            ext_stim: External stimulus for arousal
            
        Returns:
            Updated state vector [SA, SE, SP, SS]
        """
        SA, SE, SP, SS = self.state
        p = self.params
        
        # Extract metrics with defaults
        rho_def = metrics.get('rho_def', 0.0)
        R = metrics.get('R', 0.0)
        novelty = metrics.get('novelty', 0.0)
        SIR = metrics.get('SIR', 0.0)
        F = metrics.get('F', 0.0)
        
        # Compute drift terms (deterministic part)
        
        # dSA: Arousal dynamics (Eq. 2)
        drift_A = (
            p.alpha[0] * (rho_def + p.k_A * ext_stim)
            - p.gamma[0] * SA
            - p.lambda_[0] * SA**3
        )
        
        # dSE: Exploration dynamics (Eq. 3)
        R_gap = p.R_target - R
        drift_E = (
            p.alpha[1] * R_gap
            - p.gamma[1] * SE
            - p.k_AE * np.tanh(SA) * SE  # Bounded cross-coupling
            - p.lambda_[1] * SE**3
        )
        
        # dSP: Plasticity dynamics (Eq. 4)
        F_gap = p.F_target - F
        drift_P = (
            p.alpha[2] * (novelty + p.k_P * F_gap)
            - p.gamma[2] * SP
            - p.k_PS * np.tanh(SS) * SP  # Social suppression
            - p.lambda_[2] * SP**3
        )
        
        # dSS: Social dynamics (Eq. 5)
        drift_S = (
            p.alpha[3] * SIR
            - p.gamma[3] * SS
            - p.lambda_[3] * SS**3
        )
        
        drift = np.array([drift_A, drift_E, drift_P, drift_S])
        
        # Stochastic terms (Wiener process)
        noise = p.sigma * np.random.randn(4) * np.sqrt(dt)
        
        # Euler-Maruyama update
        self.state = self.state + drift * dt + noise
        
        # Store history
        self.history.append(self.state.copy())
        self.metrics_history.append(metrics.copy())
        self.current_time += dt
        self.time_history.append(self.current_time)
        
        return self.state.copy()
    
    def get_state(self) -> Dict[str, float]:
        """Get current state as dictionary"""
        return {
            'A': self.state[0],  # Arousal
            'E': self.state[1],  # Exploration
            'P': self.state[2],  # Plasticity
            'S': self.state[3]   # Social
        }
    
    def reset(self):
        """Reset controller to initial state"""
        self.state = np.zeros(4)
        self.history = []
        self.metrics_history = []
        self.time_history = []
        self.current_time = 0.0
    
    def compute_lyapunov(self) -> float:
        """
        Compute Lyapunov function V(S) = Σ(½Sᵢ² + ¼λᵢSᵢ⁴)
        For stability monitoring
        """
        S = self.state
        lam = self.params.lambda_
        V = np.sum(0.5 * S**2 + 0.25 * lam * S**4)
        return V


class MetricComputer:
    """
    Computes GSV metrics from agent state
    Uses EWMA filtering for stability
    """
    
    def __init__(self, window_size: int = 100, ewma_lambda: float = 0.05):
        self.window_size = window_size
        self.ewma_lambda = ewma_lambda
        
        # EWMA state
        self.ewma_state: Dict[str, float] = {}
        
        # Sliding windows for some metrics
        self.td_errors = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.visited_states = deque(maxlen=window_size)
        self.state_visit_counts: Dict = {}
    
    def compute_stress(self, td_error: float) -> float:
        """
        Compute stress metric ρ_def from TD error
        
        Args:
            td_error: Temporal difference error
            
        Returns:
            Stress level (0-1)
        """
        self.td_errors.append(abs(td_error))
        
        # Time-averaged absolute TD error
        if len(self.td_errors) > 0:
            current_stress = np.mean(self.td_errors)
            stress = self._ewma_update('stress', current_stress)
        else:
            stress = 0.0
        
        # Normalize to [0, 1] range (assuming TD errors < 10)
        return np.clip(stress / 10.0, 0, 1)
    
    def compute_coherence(self, policy_probs: np.ndarray) -> float:
        """
        Compute coherence R from policy entropy
        
        Args:
            policy_probs: Action probabilities from policy
            
        Returns:
            Coherence level (0-1), higher = more deterministic
        """
        # Avoid log(0)
        probs = np.clip(policy_probs, 1e-10, 1.0)
        probs = probs / probs.sum()  # Ensure normalized
        
        # Entropy: H(π) = -Σ π(a|s) log π(a|s)
        entropy = -np.sum(probs * np.log(probs))
        
        # Coherence = inverse entropy (normalized)
        max_entropy = np.log(len(probs))  # Uniform distribution
        coherence = 1.0 - (entropy / max_entropy)
        
        return self._ewma_update('coherence', coherence)
    
    def compute_novelty(self, state: int) -> float:
        """
        Compute novelty rate from state visitation
        
        Args:
            state: Current state identifier
            
        Returns:
            Novelty rate (0-1)
        """
        self.visited_states.append(state)
        
        # Count visits
        if state not in self.state_visit_counts:
            self.state_visit_counts[state] = 0
        self.state_visit_counts[state] += 1
        
        # Fraction of newly visited states in window
        if len(self.visited_states) > 0:
            recent_states = list(self.visited_states)
            new_states = sum(1 for s in recent_states 
                           if self.state_visit_counts.get(s, 0) == 1)
            novelty = new_states / len(recent_states)
        else:
            novelty = 0.0
        
        return novelty
    
    def compute_fitness(self, reward: float) -> float:
        """
        Compute fitness (average reward)
        
        Args:
            reward: Current reward
            
        Returns:
            Smoothed fitness value
        """
        self.rewards.append(reward)
        
        if len(self.rewards) > 0:
            fitness = np.mean(self.rewards)
        else:
            fitness = 0.0
        
        return self._ewma_update('fitness', fitness)
    
    def _ewma_update(self, key: str, value: float) -> float:
        """Exponentially weighted moving average"""
        if key not in self.ewma_state:
            self.ewma_state[key] = value
        else:
            self.ewma_state[key] = (
                (1 - self.ewma_lambda) * self.ewma_state[key] +
                self.ewma_lambda * value
            )
        return self.ewma_state[key]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all computed metrics"""
        return {
            'rho_def': self.ewma_state.get('stress', 0.0),
            'R': self.ewma_state.get('coherence', 0.0),
            'novelty': 0.0,  # Computed per-step
            'SIR': 0.0,  # For multi-agent (not implemented yet)
            'F': self.ewma_state.get('fitness', 0.0)
        }


class ModulationFunctions:
    """
    Modulation functions for GSV -> Fast layer parameter mapping
    All functions use bounded transformations (tanh, exp with saturation)
    """
    
    @staticmethod
    def epsilon_exploration(SE: float, eps_min: float = 0.1, eps_max: float = 0.5) -> float:
        """
        Modulate exploration rate (ε-greedy)
        
        Args:
            SE: Exploration component of GSV
            eps_min: Minimum exploration rate
            eps_max: Maximum exploration rate
            
        Returns:
            Exploration rate ∈ [eps_min, eps_max]
        """
        return eps_min + (eps_max - eps_min) * 0.5 * (1 + np.tanh(SE))
    
    @staticmethod
    def alpha_learning(SP: float, alpha_base: float = 0.001, k_alpha: float = 0.5) -> float:
        """
        Modulate learning rate
        
        Args:
            SP: Plasticity component of GSV
            alpha_base: Base learning rate
            k_alpha: Sensitivity parameter
            
        Returns:
            Modulated learning rate
        """
        return alpha_base * np.exp(k_alpha * SP / (1 + abs(SP)))
    
    @staticmethod
    def gamma_discount(SA: float, gamma_base: float = 0.99, k_gamma: float = 0.1) -> float:
        """
        Modulate discount factor (urgency under stress)
        
        Args:
            SA: Arousal component of GSV
            gamma_base: Base discount factor
            k_gamma: Sensitivity (must be in [0, 1])
            
        Returns:
            Modulated discount factor
        """
        k_gamma = np.clip(k_gamma, 0, 1)  # Ensure non-negative gamma
        return gamma_base * (1 - k_gamma * 0.5 * (1 + np.tanh(SA)))
    
    @staticmethod
    def temperature(SE: float, T_min: float = 0.5, T_max: float = 2.0) -> float:
        """
        Modulate sampling temperature (for LLMs or softmax policies)
        
        Args:
            SE: Exploration component of GSV
            T_min: Minimum temperature
            T_max: Maximum temperature
            
        Returns:
            Temperature ∈ [T_min, T_max]
        """
        return T_min + (T_max - T_min) * 0.5 * (1 + np.tanh(SE))
    
    @staticmethod
    def trust_weight(SS: float, w0: float = 0.5, k_w: float = 0.5) -> float:
        """
        Modulate social reward weight
        
        Args:
            SS: Social component of GSV
            w0: Base weight
            k_w: Sensitivity
            
        Returns:
            Social reward weight
        """
        return w0 * 0.5 * (1 + np.tanh(SS))


# Example usage demonstration
if __name__ == "__main__":
    print("GSV 2.0 Core Implementation")
    print("=" * 50)
    
    # Initialize controller with balanced parameters
    params = GSV2Params()
    gsv = GSV2Controller(params)
    
    print(f"\nInitial state: {gsv.get_state()}")
    
    # Simulate a stress scenario
    print("\nSimulating stress response...")
    metrics = {
        'rho_def': 0.8,  # High stress
        'R': 0.3,        # Low coherence
        'novelty': 0.2,
        'SIR': 0.0,
        'F': -5.0
    }
    
    for i in range(10):
        gsv.step(metrics, dt=1.0)
        if i % 3 == 0:
            state = gsv.get_state()
            print(f"  Step {i}: SA={state['A']:.3f}, SE={state['E']:.3f}, "
                  f"SP={state['P']:.3f}, SS={state['S']:.3f}")
    
    # Test modulation functions
    print("\n" + "=" * 50)
    print("Modulation Functions Test:")
    state = gsv.get_state()
    
    epsilon = ModulationFunctions.epsilon_exploration(state['E'])
    alpha = ModulationFunctions.alpha_learning(state['P'])
    gamma = ModulationFunctions.gamma_discount(state['A'])
    
    print(f"  Exploration rate (ε): {epsilon:.3f}")
    print(f"  Learning rate (α): {alpha:.6f}")
    print(f"  Discount factor (γ): {gamma:.3f}")
    
    # Test metric computer
    print("\n" + "=" * 50)
    print("Metric Computer Test:")
    computer = MetricComputer()
    
    # Simulate some agent steps
    for _ in range(5):
        td_error = np.random.randn() * 2
        policy = np.random.dirichlet([1, 1, 1, 1])
        state_id = np.random.randint(0, 10)
        reward = np.random.randn()
        
        stress = computer.compute_stress(td_error)
        coherence = computer.compute_coherence(policy)
        novelty = computer.compute_novelty(state_id)
        fitness = computer.compute_fitness(reward)
    
    metrics = computer.get_metrics()
    print(f"  Computed metrics: {metrics}")
    
    print("\n" + "=" * 50)
    print("GSV 2.0 Core: ✓ Implementation complete")