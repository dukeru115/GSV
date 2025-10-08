"""
GSV 2.0 - Analysis and Diagnostics Module
Provides tools for:
- Stability analysis (Jacobian eigenvalues, Lyapunov function)
- Phase space analysis
- Parameter sensitivity
- System diagnostics and monitoring
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import linalg


class StabilityAnalyzer:
    """
    Analyzes stability properties of GSV 2.0 system
    Based on Theorem 1 and Lyapunov analysis from paper (Sections 4.1, Appendix B)
    """
    
    def __init__(self, gsv_controller):
        self.gsv = gsv_controller
        self.params = gsv_controller.params
    
    def check_stability_conditions(self) -> Dict[str, bool]:
        """
        Check analytical stability conditions from Theorem 1
        
        Returns:
            Dictionary with condition checks
        """
        p = self.params
        
        results = {
            'gamma_E_condition': p.gamma[1] > p.k_AE,
            'gamma_P_condition': p.gamma[2] > p.k_PS,
            'all_gamma_positive': np.all(p.gamma > 0),
            'all_lambda_positive': np.all(p.lambda_ > 0),
            'all_sigma_nonnegative': np.all(p.sigma >= 0)
        }
        
        results['overall_stable'] = all(results.values())
        
        return results
    
    def compute_jacobian(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Jacobian matrix at given state
        
        For GSV 2.0, the Jacobian is:
        J = [[J11,   0,   0,   0],
             [J21, J22,   0,   0],
             [  0,   0, J33, J34],
             [  0,   0,   0, J44]]
        
        Args:
            state: State vector [SA, SE, SP, SS], defaults to current state
            
        Returns:
            4×4 Jacobian matrix
        """
        if state is None:
            state = self.gsv.state
        
        SA, SE, SP, SS = state
        p = self.params
        
        # Diagonal elements
        J11 = -p.gamma[0] - 3 * p.lambda_[0] * SA**2
        J22 = -p.gamma[1] - p.k_AE * np.tanh(SA) - 3 * p.lambda_[1] * SE**2
        J33 = -p.gamma[2] - p.k_PS * np.tanh(SS) - 3 * p.lambda_[2] * SP**2
        J44 = -p.gamma[3] - 3 * p.lambda_[3] * SS**2
        
        # Off-diagonal coupling terms
        # J21: derivative of SE dynamics w.r.t. SA
        J21 = -p.k_AE * (1 - np.tanh(SA)**2) * SE  # sech²(SA) * SE
        
        # J34: derivative of SP dynamics w.r.t. SS
        J34 = -p.k_PS * (1 - np.tanh(SS)**2) * SP  # sech²(SS) * SP
        
        # Construct Jacobian
        J = np.array([
            [J11,   0,   0,   0],
            [J21, J22,   0,   0],
            [  0,   0, J33, J34],
            [  0,   0,   0, J44]
        ])
        
        return J
    
    def compute_eigenvalues(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute eigenvalues of Jacobian
        
        For block-triangular structure, eigenvalues are diagonal elements
        """
        J = self.compute_jacobian(state)
        eigenvalues = np.linalg.eigvals(J)
        return eigenvalues
    
    def is_locally_stable(self, state: Optional[np.ndarray] = None) -> bool:
        """
        Check local stability at given state
        System is stable if all eigenvalues have negative real parts
        """
        eigenvalues = self.compute_eigenvalues(state)
        return np.all(np.real(eigenvalues) < 0)
    
    def compute_lyapunov_function(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute Lyapunov function V(S) = Σ(½Sᵢ² + ¼λᵢSᵢ⁴)
        
        Lower values indicate states closer to origin (equilibrium)
        """
        if state is None:
            state = self.gsv.state
        
        p = self.params
        V = np.sum(0.5 * state**2 + 0.25 * p.lambda_ * state**4)
        return V
    
    def compute_lyapunov_derivative(self, state: Optional[np.ndarray] = None) -> float:
        """
        Compute time derivative of Lyapunov function along trajectory
        
        For stability, dV/dt should be negative
        """
        if state is None:
            state = self.gsv.state
        
        p = self.params
        
        # Simplified computation (ignoring cross-terms for approximation)
        dV_dt = -np.sum(p.gamma * state**2) - np.sum(p.lambda_ * state**4)
        
        return dV_dt
    
    def analyze_trajectory(self, history: List[np.ndarray]) -> Dict:
        """
        Analyze stability properties over entire trajectory
        
        Args:
            history: List of state vectors
            
        Returns:
            Dictionary with trajectory analysis
        """
        n_points = len(history)
        
        eigenvalues_real = np.zeros((n_points, 4))
        lyapunov_values = np.zeros(n_points)
        stable_points = []
        
        for i, state in enumerate(history):
            eigs = self.compute_eigenvalues(state)
            eigenvalues_real[i] = np.real(eigs)
            lyapunov_values[i] = self.compute_lyapunov_function(state)
            stable_points.append(self.is_locally_stable(state))
        
        return {
            'eigenvalues_real': eigenvalues_real,
            'lyapunov_values': lyapunov_values,
            'stable_points': stable_points,
            'fraction_stable': np.mean(stable_points),
            'max_lyapunov': np.max(lyapunov_values),
            'min_eigenvalue': np.min(eigenvalues_real),
            'max_eigenvalue': np.max(eigenvalues_real)
        }


class PhaseSpaceAnalyzer:
    """
    Analyzes phase space structure and behavioral regimes
    Based on Section 4.2 of paper
    """
    
    def __init__(self, gsv_controller):
        self.gsv = gsv_controller
    
    def identify_regime(self, state: Optional[np.ndarray] = None) -> str:
        """
        Identify behavioral regime based on GSV state
        
        Regimes from Table 4.2.2:
        - Explorer: [0.5, +1, +1, 0]
        - Guardian: [+1, -1, -0.5, 0]
        - Collaborator: [0, 0, -0.5, +1]
        - Adapter: [0, +0.5, +1, -0.5]
        """
        if state is None:
            state = self.gsv.state
        
        SA, SE, SP, SS = state
        
        # Define regime prototypes
        regimes = {
            'Explorer': np.array([0.5, 1.0, 1.0, 0.0]),
            'Guardian': np.array([1.0, -1.0, -0.5, 0.0]),
            'Collaborator': np.array([0.0, 0.0, -0.5, 1.0]),
            'Adapter': np.array([0.0, 0.5, 1.0, -0.5]),
            'Baseline': np.array([0.0, 0.0, 0.0, 0.0])
        }
        
        # Find closest regime by Euclidean distance
        min_dist = float('inf')
        closest_regime = 'Unknown'
        
        for regime_name, prototype in regimes.items():
            dist = np.linalg.norm(state - prototype)
            if dist < min_dist:
                min_dist = dist
                closest_regime = regime_name
        
        return closest_regime
    
    def compute_attractor_basin(
        self,
        axis1: int = 0,
        axis2: int = 1,
        grid_size: int = 30,
        range_: Tuple[float, float] = (-2, 2)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute vector field for phase portrait
        
        Args:
            axis1, axis2: Which GSV components to plot (0=SA, 1=SE, 2=SP, 3=SS)
            grid_size: Resolution of grid
            range_: (min, max) for both axes
            
        Returns:
            X, Y meshgrid and vector field U, V
        """
        x = np.linspace(range_[0], range_[1], grid_size)
        y = np.linspace(range_[0], range_[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        p = self.gsv.params
        
        # Compute vector field at each point
        for i in range(grid_size):
            for j in range(grid_size):
                state = np.zeros(4)
                state[axis1] = X[i, j]
                state[axis2] = Y[i, j]
                
                # Compute drift (simplified, without metrics)
                SA, SE, SP, SS = state
                
                drift = np.zeros(4)
                drift[0] = -p.gamma[0] * SA - p.lambda_[0] * SA**3
                drift[1] = -p.gamma[1] * SE - p.k_AE * np.tanh(SA) * SE - p.lambda_[1] * SE**3
                drift[2] = -p.gamma[2] * SP - p.k_PS * np.tanh(SS) * SP - p.lambda_[2] * SP**3
                drift[3] = -p.gamma[3] * SS - p.lambda_[3] * SS**3
                
                U[i, j] = drift[axis1]
                V[i, j] = drift[axis2]
        
        return X, Y, U, V


class DiagnosticMonitor:
    """
    Real-time monitoring and diagnostics for GSV system
    """
    
    def __init__(self, gsv_controller, warning_thresholds: Optional[Dict] = None):
        self.gsv = gsv_controller
        self.analyzer = StabilityAnalyzer(gsv_controller)
        
        # Default warning thresholds
        default_thresholds = {
            'max_state_norm': 3.0,
            'max_lyapunov': 10.0,
            'min_eigenvalue': -1.0,
            'max_td_error': 50.0
        }
        self.thresholds = warning_thresholds or default_thresholds
        
        self.warnings: List[Dict] = []
    
    def check_health(self) -> Dict:
        """
        Comprehensive health check of GSV system
        
        Returns:
            Dictionary with status and any warnings
        """
        state = self.gsv.state
        status = {
            'timestamp': self.gsv.current_time,
            'state_norm': np.linalg.norm(state),
            'lyapunov': self.analyzer.compute_lyapunov_function(),
            'is_stable': self.analyzer.is_locally_stable(),
            'warnings': []
        }
        
        # Check state norm
        if status['state_norm'] > self.thresholds['max_state_norm']:
            warning = {
                'type': 'HIGH_STATE_NORM',
                'value': status['state_norm'],
                'threshold': self.thresholds['max_state_norm'],
                'message': 'State vector norm exceeds threshold'
            }
            status['warnings'].append(warning)
            self.warnings.append(warning)
        
        # Check Lyapunov function
        if status['lyapunov'] > self.thresholds['max_lyapunov']:
            warning = {
                'type': 'HIGH_LYAPUNOV',
                'value': status['lyapunov'],
                'threshold': self.thresholds['max_lyapunov'],
                'message': 'Lyapunov function indicates divergence'
            }
            status['warnings'].append(warning)
            self.warnings.append(warning)
        
        # Check stability
        if not status['is_stable']:
            eigenvalues = self.analyzer.compute_eigenvalues()
            warning = {
                'type': 'UNSTABLE_EQUILIBRIUM',
                'eigenvalues': eigenvalues.tolist(),
                'message': 'System is locally unstable'
            }
            status['warnings'].append(warning)
            self.warnings.append(warning)
        
        status['health'] = 'HEALTHY' if len(status['warnings']) == 0 else 'WARNING'
        
        return status
    
    def print_status(self):
        """Print current system status"""
        status = self.check_health()
        
        print("\n" + "="*60)
        print("GSV 2.0 SYSTEM STATUS")
        print("="*60)
        print(f"Time: {status['timestamp']:.1f}s")
        print(f"State: {self.gsv.state}")
        print(f"State Norm: {status['state_norm']:.3f}")
        print(f"Lyapunov V(S): {status['lyapunov']:.3f}")
        print(f"Locally Stable: {status['is_stable']}")
        print(f"Health: {status['health']}")
        
        if status['warnings']:
            print("\n⚠️  WARNINGS:")
            for w in status['warnings']:
                print(f"  - {w['type']}: {w['message']}")
        
        print("="*60)


def visualize_stability_analysis(gsv_controller, history: List[np.ndarray]):
    """
    Create comprehensive stability visualization
    """
    analyzer = StabilityAnalyzer(gsv_controller)
    analysis = analyzer.analyze_trajectory(history)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Eigenvalues over time
    ax = axes[0, 0]
    time = np.arange(len(history))
    for i in range(4):
        ax.plot(time, analysis['eigenvalues_real'][:, i], 
               label=f'λ{i+1}', linewidth=2)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Real(λ)')
    ax.set_title('Jacobian Eigenvalues Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Lyapunov function
    ax = axes[0, 1]
    ax.plot(time, analysis['lyapunov_values'], linewidth=2, color='purple')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('V(S)')
    ax.set_title('Lyapunov Function')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: State norm
    ax = axes[1, 0]
    norms = [np.linalg.norm(s) for s in history]
    ax.plot(time, norms, linewidth=2, color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('||S||')
    ax.set_title('State Vector Norm')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Stability indicator
    ax = axes[1, 1]
    stable_indicator = [1 if s else 0 for s in analysis['stable_points']]
    ax.plot(time, stable_indicator, linewidth=2, color='orange')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stable (1) / Unstable (0)')
    ax.set_title(f'Local Stability (Fraction: {analysis["fraction_stable"]:.2%})')
    ax.set_ylim([-0.1, 1.1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_phase_space(gsv_controller, history: List[np.ndarray]):
    """
    Visualize phase space with vector field and trajectory
    """
    phase_analyzer = PhaseSpaceAnalyzer(gsv_controller)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SA-SE plane
    ax = axes[0]
    X, Y, U, V = phase_analyzer.compute_attractor_basin(axis1=0, axis2=1)
    
    # Plot vector field
    ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], 
             alpha=0.5, color='gray')
    
    # Plot trajectory
    history_array = np.array(history)
    ax.plot(history_array[:, 0], history_array[:, 1], 
           'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(history_array[0, 0], history_array[0, 1], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(history_array[-1, 0], history_array[-1, 1], 
              c='red', s=100, marker='s', label='End', zorder=5)
    
    ax.set_xlabel('SA (Arousal)')
    ax.set_ylabel('SE (Exploration)')
    ax.set_title('SA-SE Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SP-SS plane
    ax = axes[1]
    X, Y, U, V = phase_analyzer.compute_attractor_basin(axis1=2, axis2=3)
    
    ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], 
             alpha=0.5, color='gray')
    ax.plot(history_array[:, 2], history_array[:, 3], 
           'b-', linewidth=2, alpha=0.7, label='Trajectory')
    ax.scatter(history_array[0, 2], history_array[0, 3], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(history_array[-1, 2], history_array[-1, 3], 
              c='red', s=100, marker='s', label='End', zorder=5)
    
    ax.set_xlabel('SP (Plasticity)')
    ax.set_ylabel('SS (Social)')
    ax.set_title('SP-SS Phase Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Demo
if __name__ == "__main__":
    from gsv2_core import GSV2Controller, GSV2Params
    
    print("GSV 2.0 Analysis & Diagnostics Demo")
    print("=" * 60)
    
    # Create controller
    gsv = GSV2Controller(GSV2Params())
    analyzer = StabilityAnalyzer(gsv)
    monitor = DiagnosticMonitor(gsv)
    
    # Check initial stability conditions
    print("\n1. Stability Conditions Check:")
    conditions = analyzer.check_stability_conditions()
    for condition, result in conditions.items():
        status = "✓" if result else "✗"
        print(f"  {status} {condition}: {result}")
    
    # Simulate some dynamics
    print("\n2. Simulating dynamics...")
    metrics = {
        'rho_def': 0.5,
        'R': 0.6,
        'novelty': 0.3,
        'SIR': 0.0,
        'F': 2.0
    }
    
    for _ in range(50):
        gsv.step(metrics, dt=1.0)
    
    # Check health
    print("\n3. Health Check:")
    monitor.print_status()
    
    # Analyze trajectory
    print("\n4. Trajectory Analysis:")
    traj_analysis = analyzer.analyze_trajectory(gsv.history)
    print(f"  Fraction stable: {traj_analysis['fraction_stable']:.2%}")
    print(f"  Max Lyapunov: {traj_analysis['max_lyapunov']:.3f}")
    print(f"  Eigenvalue range: [{traj_analysis['min_eigenvalue']:.3f}, "
          f"{traj_analysis['max_eigenvalue']:.3f}]")
    
    # Identify current regime
    phase_analyzer = PhaseSpaceAnalyzer(gsv)
    regime = phase_analyzer.identify_regime()
    print(f"\n5. Current Behavioral Regime: {regime}")
    
    print("\n" + "=" * 60)
    print("GSV 2.0 Analysis: ✓ Complete")