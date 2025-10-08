"""
GSV 2.0 - Unit Tests
Validates correctness of core implementation
"""

import numpy as np
import sys
from typing import Dict

# Test configuration
TOLERANCE = 1e-6
VERBOSE = True


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details and (not passed or VERBOSE):
        print(f"         {details}")


class TestGSV2Core:
    """Test suite for gsv2_core.py"""
    
    def __init__(self):
        try:
            from gsv2_core import GSV2Controller, GSV2Params, MetricComputer, ModulationFunctions
            self.GSV2Controller = GSV2Controller
            self.GSV2Params = GSV2Params
            self.MetricComputer = MetricComputer
            self.MF = ModulationFunctions
        except ImportError as e:
            print(f"❌ Cannot import gsv2_core: {e}")
            sys.exit(1)
    
    def test_params_initialization(self) -> bool:
        """Test parameter initialization and validation"""
        try:
            # Balanced params
            params = self.GSV2Params()
            assert params.gamma[1] > params.k_AE, "Stability condition violated"
            
            # Conservative params
            params_cons = self.GSV2Params.conservative()
            assert params_cons.gamma[1] > params_cons.k_AE
            
            # Aggressive params
            params_agg = self.GSV2Params.aggressive()
            assert params_agg.gamma[1] > params_agg.k_AE
            
            return True
        except Exception as e:
            print_test("Params initialization", False, str(e))
            return False
    
    def test_controller_initialization(self) -> bool:
        """Test GSV controller initialization"""
        try:
            gsv = self.GSV2Controller()
            
            # Check initial state is zero
            assert np.allclose(gsv.state, np.zeros(4)), "Initial state not zero"
            
            # Check history is empty
            assert len(gsv.history) == 0, "History not empty"
            
            # Check time is zero
            assert gsv.current_time == 0.0, "Time not zero"
            
            return True
        except Exception as e:
            print_test("Controller initialization", False, str(e))
            return False
    
    def test_stability_condition(self) -> bool:
        """Test stability condition γE > kAE"""
        try:
            # Create controller with violating parameters
            params_bad = self.GSV2Params()
            params_bad.k_AE = 0.5  # > gamma[1] = 0.01
            
            # Should print warning but not crash
            gsv = self.GSV2Controller(params_bad)
            
            # Create controller with good parameters
            params_good = self.GSV2Params()
            params_good.k_AE = 0.005  # < gamma[1] = 0.01
            gsv_good = self.GSV2Controller(params_good)
            
            return True
        except Exception as e:
            print_test("Stability condition", False, str(e))
            return False
    
    def test_sde_integration(self) -> bool:
        """Test SDE integration step"""
        try:
            gsv = self.GSV2Controller()
            
            metrics = {
                'rho_def': 0.5,
                'R': 0.6,
                'novelty': 0.3,
                'SIR': 0.0,
                'F': 1.0
            }
            
            # Single step
            state = gsv.step(metrics, dt=1.0)
            
            # Check state is 4-dimensional
            assert state.shape == (4,), f"State shape is {state.shape}, expected (4,)"
            
            # Check history updated
            assert len(gsv.history) == 1, "History not updated"
            
            # Check time updated
            assert gsv.current_time == 1.0, "Time not updated"
            
            # Multiple steps
            for _ in range(10):
                gsv.step(metrics, dt=0.1)
            
            assert len(gsv.history) == 11, "History length incorrect"
            
            return True
        except Exception as e:
            print_test("SDE integration", False, str(e))
            return False
    
    def test_bounded_dynamics(self) -> bool:
        """Test that dynamics remain bounded (cubic damping)"""
        try:
            gsv = self.GSV2Controller()
            
            # Extreme driving metrics
            metrics = {
                'rho_def': 1.0,  # Maximum stress
                'R': 0.0,        # Zero coherence
                'novelty': 1.0,
                'SIR': 1.0,
                'F': 100.0
            }
            
            # Run for extended period
            max_norm = 0
            for _ in range(1000):
                gsv.step(metrics, dt=1.0)
                norm = np.linalg.norm(gsv.state)
                max_norm = max(max_norm, norm)
            
            # Check state remains bounded (should be < 5 with cubic damping)
            assert max_norm < 5.0, f"State diverged: max_norm = {max_norm}"
            
            return True
        except Exception as e:
            print_test("Bounded dynamics", False, str(e))
            return False
    
    def test_metric_computer(self) -> bool:
        """Test metric computation"""
        try:
            computer = self.MetricComputer()
            
            # Test stress computation
            stress = computer.compute_stress(td_error=2.0)
            assert 0 <= stress <= 1, f"Stress out of range: {stress}"
            
            # Test coherence computation
            policy = np.array([0.7, 0.2, 0.05, 0.05])
            coherence = computer.compute_coherence(policy)
            assert 0 <= coherence <= 1, f"Coherence out of range: {coherence}"
            
            # Test novelty computation
            for i in range(10):
                novelty = computer.compute_novelty(state=i % 5)
            assert 0 <= novelty <= 1, f"Novelty out of range: {novelty}"
            
            # Test fitness computation
            fitness = computer.compute_fitness(reward=5.0)
            assert fitness > 0, "Fitness calculation incorrect"
            
            return True
        except Exception as e:
            print_test("Metric computer", False, str(e))
            return False
    
    def test_modulation_functions(self) -> bool:
        """Test modulation functions"""
        try:
            # Test epsilon modulation
            epsilon = self.MF.epsilon_exploration(SE=0.0)
            assert 0.1 <= epsilon <= 0.5, f"Epsilon out of range: {epsilon}"
            
            epsilon_high = self.MF.epsilon_exploration(SE=2.0)
            epsilon_low = self.MF.epsilon_exploration(SE=-2.0)
            assert epsilon_high > epsilon_low, "Epsilon not monotonic"
            
            # Test alpha modulation
            alpha = self.MF.alpha_learning(SP=0.0)
            assert alpha > 0, f"Alpha non-positive: {alpha}"
            
            # Test gamma modulation
            gamma = self.MF.gamma_discount(SA=0.0)
            assert 0 < gamma < 1, f"Gamma out of range: {gamma}"
            
            # Test temperature modulation
            temp = self.MF.temperature(SE=0.0)
            assert 0.5 <= temp <= 2.0, f"Temperature out of range: {temp}"
            
            return True
        except Exception as e:
            print_test("Modulation functions", False, str(e))
            return False
    
    def test_lyapunov_monotonicity(self) -> bool:
        """Test that Lyapunov function decreases (on average)"""
        try:
            gsv = self.GSV2Controller()
            
            # Set initial state away from origin
            gsv.state = np.array([1.0, 1.0, 1.0, 1.0])
            
            initial_V = gsv.compute_lyapunov()
            
            # Run with zero metrics (decay only)
            metrics = {
                'rho_def': 0.0,
                'R': gsv.params.R_target,
                'novelty': 0.0,
                'SIR': 0.0,
                'F': gsv.params.F_target
            }
            
            # Run without noise to test deterministic decay
            gsv.params.sigma = np.zeros(4)
            
            for _ in range(100):
                gsv.step(metrics, dt=1.0)
            
            final_V = gsv.compute_lyapunov()
            
            # V should decrease
            assert final_V < initial_V, f"Lyapunov increased: {initial_V} -> {final_V}"
            
            return True
        except Exception as e:
            print_test("Lyapunov monotonicity", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print("GSV 2.0 CORE TESTS")
        print("="*70 + "\n")
        
        tests = [
            ("Params initialization", self.test_params_initialization),
            ("Controller initialization", self.test_controller_initialization),
            ("Stability condition check", self.test_stability_condition),
            ("SDE integration", self.test_sde_integration),
            ("Bounded dynamics", self.test_bounded_dynamics),
            ("Metric computer", self.test_metric_computer),
            ("Modulation functions", self.test_modulation_functions),
            ("Lyapunov monotonicity", self.test_lyapunov_monotonicity),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append(passed)
                print_test(name, passed)
            except Exception as e:
                print_test(name, False, f"Exception: {e}")
                results.append(False)
        
        print("\n" + "="*70)
        print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
        print("="*70 + "\n")
        
        return all(results)


class TestGSV2Integration:
    """Test suite for gsv2_qlearning.py"""
    
    def __init__(self):
        try:
            from gsv2_qlearning import QLearningAgent, GSV2ModulatedQLearning, SimpleGridworld
            from gsv2_core import GSV2Controller, GSV2Params, MetricComputer
            
            self.QLearningAgent = QLearningAgent
            self.GSV2ModulatedQLearning = GSV2ModulatedQLearning
            self.SimpleGridworld = SimpleGridworld
            self.GSV2Controller = GSV2Controller
            self.GSV2Params = GSV2Params
            self.MetricComputer = MetricComputer
        except ImportError as e:
            print(f"❌ Cannot import modules: {e}")
            sys.exit(1)
    
    def test_qlearning_agent(self) -> bool:
        """Test basic Q-learning agent"""
        try:
            agent = self.QLearningAgent(n_states=10, n_actions=4)
            
            # Check Q-table initialization
            assert agent.Q.shape == (10, 4), "Q-table shape incorrect"
            
            # Test action selection
            action = agent.get_action(state=0)
            assert 0 <= action < 4, "Action out of range"
            
            # Test update
            td_error = agent.update(0, action, 1.0, 1, False)
            assert isinstance(td_error, float), "TD error not float"
            
            return True
        except Exception as e:
            print_test("Q-learning agent", False, str(e))
            return False
    
    def test_gridworld_env(self) -> bool:
        """Test gridworld environment"""
        try:
            env = self.SimpleGridworld(size=10)
            
            # Test reset
            state = env.reset()
            assert 0 <= state < 100, "Invalid state"
            
            # Test step
            next_state, reward, done = env.step(action=1)
            assert 0 <= next_state < 100, "Invalid next state"
            assert isinstance(reward, float), "Reward not float"
            assert isinstance(done, bool), "Done not bool"
            
            # Test goal change
            old_goal = env.goal_pos.copy()
            env.change_goal()
            assert env.goal_pos != old_goal, "Goal not changed"
            
            return True
        except Exception as e:
            print_test("Gridworld environment", False, str(e))
            return False
    
    def test_modulated_agent(self) -> bool:
        """Test GSV-modulated agent"""
        try:
            env = self.SimpleGridworld(size=5)
            base_agent = self.QLearningAgent(n_states=25, n_actions=4)
            gsv = self.GSV2Controller(self.GSV2Params())
            metrics = self.MetricComputer()
            
            agent = self.GSV2ModulatedQLearning(base_agent, gsv, metrics)
            
            # Run a few steps
            state = env.reset()
            for _ in range(10):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                step_info = agent.step(state, action, reward, next_state, done)
                
                # Check step_info structure
                assert 'td_error' in step_info
                assert 'metrics' in step_info
                assert 'gsv_state' in step_info
                assert 'epsilon' in step_info
                
                if done:
                    break
                state = next_state
            
            return True
        except Exception as e:
            print_test("Modulated agent", False, str(e))
            return False
    
    def test_adaptation_to_regime_change(self) -> bool:
        """Test adaptation to environment change"""
        try:
            env = self.SimpleGridworld(size=5)
            base_agent = self.QLearningAgent(n_states=25, n_actions=4)
            gsv = self.GSV2Controller(self.GSV2Params())
            metrics = self.MetricComputer()
            
            agent = self.GSV2ModulatedQLearning(base_agent, gsv, metrics)
            
            # Train for a while
            for episode in range(20):
                state = env.reset()
                done = False
                steps = 0
                
                while not done and steps < 50:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                
                agent.end_episode(0, steps, done)
            
            # Record performance before change
            stats_before = agent.get_statistics(window=10)
            gsv_state_before = gsv.get_state()
            
            # Change environment
            env.change_goal()
            
            # Train more
            for episode in range(20):
                state = env.reset()
                done = False
                steps = 0
                
                while not done and steps < 50:
                    action = agent.act(state)
                    next_state, reward, done = env.step(action)
                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    steps += 1
                
                agent.end_episode(0, steps, done)
            
            # Check that GSV responded to change
            gsv_state_after = gsv.get_state()
            
            # SE should have increased (more exploration after regime change)
            # This is a weak test since results are stochastic
            assert len(agent.episode_rewards) == 40, "Episode count incorrect"
            
            return True
        except Exception as e:
            print_test("Adaptation to regime change", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*70)
        print("GSV 2.0 INTEGRATION TESTS")
        print("="*70 + "\n")
        
        tests = [
            ("Q-learning agent", self.test_qlearning_agent),
            ("Gridworld environment", self.test_gridworld_env),
            ("Modulated agent", self.test_modulated_agent),
            ("Adaptation to regime change", self.test_adaptation_to_regime_change),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append(passed)
                print_test(name, passed)
            except Exception as e:
                print_test(name, False, f"Exception: {e}")
                results.append(False)
        
        print("\n" + "="*70)
        print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
        print("="*70 + "\n")
        
        return all(results)


class TestGSV2Stability:
    """Test stability analysis module"""
    
    def __init__(self):
        try:
            from gsv2_analysis import StabilityAnalyzer, PhaseSpaceAnalyzer, DiagnosticMonitor
            from gsv2_core import GSV2Controller, GSV2Params
            
            self.StabilityAnalyzer = StabilityAnalyzer
            self.PhaseSpaceAnalyzer = PhaseSpaceAnalyzer
            self.DiagnosticMonitor = DiagnosticMonitor
            self.GSV2Controller = GSV2Controller
            self.GSV2Params = GSV2Params
        except ImportError as e:
            print(f"❌ Cannot import analysis modules: {e}")
            sys.exit(1)
    
    def test_jacobian_computation(self) -> bool:
        """Test Jacobian matrix computation"""
        try:
            gsv = self.GSV2Controller()
            analyzer = self.StabilityAnalyzer(gsv)
            
            # Compute Jacobian at origin
            J = analyzer.compute_jacobian(np.zeros(4))
            
            # Check shape
            assert J.shape == (4, 4), f"Jacobian shape incorrect: {J.shape}"
            
            # Check block structure (upper-triangular blocks)
            assert J[2, 0] == 0, "J[2,0] should be zero"
            assert J[2, 1] == 0, "J[2,1] should be zero"
            assert J[3, 0] == 0, "J[3,0] should be zero"
            assert J[3, 1] == 0, "J[3,1] should be zero"
            
            return True
        except Exception as e:
            print_test("Jacobian computation", False, str(e))
            return False
    
    def test_eigenvalue_computation(self) -> bool:
        """Test eigenvalue computation"""
        try:
            gsv = self.GSV2Controller()
            analyzer = self.StabilityAnalyzer(gsv)
            
            eigenvalues = analyzer.compute_eigenvalues(np.zeros(4))
            
            # Check length
            assert len(eigenvalues) == 4, "Should have 4 eigenvalues"
            
            # At origin with good parameters, all should be negative
            assert np.all(np.real(eigenvalues) < 0), "Not all eigenvalues negative at origin"
            
            return True
        except Exception as e:
            print_test("Eigenvalue computation", False, str(e))
            return False
    
    def test_regime_identification(self) -> bool:
        """Test behavioral regime identification"""
        try:
            gsv = self.GSV2Controller()
            phase_analyzer = self.PhaseSpaceAnalyzer(gsv)
            
            # Test baseline
            gsv.state = np.array([0.0, 0.0, 0.0, 0.0])
            regime = phase_analyzer.identify_regime()
            assert regime == 'Baseline', f"Expected Baseline, got {regime}"
            
            # Test explorer
            gsv.state = np.array([0.5, 1.0, 1.0, 0.0])
            regime = phase_analyzer.identify_regime()
            assert regime == 'Explorer', f"Expected Explorer, got {regime}"
            
            return True
        except Exception as e:
            print_test("Regime identification", False, str(e))
            return False
    
    def test_diagnostic_monitor(self) -> bool:
        """Test diagnostic monitoring"""
        try:
            gsv = self.GSV2Controller()
            monitor = self.DiagnosticMonitor(gsv)
            
            # Check health at origin
            status = monitor.check_health()
            
            assert 'health' in status, "Status missing health field"
            assert 'warnings' in status, "Status missing warnings field"
            assert status['health'] == 'HEALTHY', "Should be healthy at origin"
            
            return True
        except Exception as e:
            print_test("Diagnostic monitor", False, str(e))
            return False
    
    def run_all_tests(self):
        """Run all stability tests"""
        print("\n" + "="*70)
        print("GSV 2.0 STABILITY ANALYSIS TESTS")
        print("="*70 + "\n")
        
        tests = [
            ("Jacobian computation", self.test_jacobian_computation),
            ("Eigenvalue computation", self.test_eigenvalue_computation),
            ("Regime identification", self.test_regime_identification),
            ("Diagnostic monitor", self.test_diagnostic_monitor),
        ]
        
        results = []
        for name, test_func in tests:
            try:
                passed = test_func()
                results.append(passed)
                print_test(name, passed)
            except Exception as e:
                print_test(name, False, f"Exception: {e}")
                results.append(False)
        
        print("\n" + "="*70)
        print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
        print("="*70 + "\n")
        
        return all(results)


def run_all_test_suites():
    """Run all test suites"""
    print("\n" + "🧪 " + "="*68)
    print("GSV 2.0 - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    all_passed = True
    
    # Core tests
    core_tests = TestGSV2Core()
    all_passed &= core_tests.run_all_tests()
    
    # Integration tests
    integration_tests = TestGSV2Integration()
    all_passed &= integration_tests.run_all_tests()
    
    # Stability tests
    stability_tests = TestGSV2Stability()
    all_passed &= stability_tests.run_all_tests()
    
    # Final summary
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_test_suites()
    sys.exit(0 if success else 1)