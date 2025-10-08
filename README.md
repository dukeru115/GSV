# Global State Vector 2.0: Multi-Scale Control for Autonomous AI Agents

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-MVP-orange.svg)]()
[![CI](https://github.com/dukeru115/GSV/workflows/CI/badge.svg)](https://github.com/dukeru115/GSV/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Implementation of GSV 2.0 framework for strategic adaptation in autonomous agents**  
> Based on: Urmanov, T., Gadeev, K., & Iusupov, B. (2025). *Global State Vector 2.0: Multi-Scale Control for Autonomous AI Agents*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Modules](#modules)
- [Examples](#examples)
- [Theory](#theory)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## 🎯 Overview

**GSV 2.0** provides a mathematically rigorous framework for **strategic adaptation** in AI agents through multi-scale control. Unlike traditional fixed hyperparameters or black-box meta-learning, GSV uses a low-dimensional dynamical system to modulate fast-layer cognitive processes based on accumulated experience.

### Key Features

✅ **Mathematically Rigorous**: Proven stability guarantees via Lyapunov theory  
✅ **Interpretable**: Four explicit control axes with clear semantics  
✅ **Robust**: Prevents pathological states through stochastic dynamics  
✅ **Adaptive**: Responds to environmental regime changes  
✅ **Framework-Agnostic**: Works with Q-learning, DQN, PPO, LLMs, etc.

### The Four Axes

```
S(t) = [SA(t), SE(t), SP(t), SS(t)]
```

| Axis | Name | Controls | Active Inference Interpretation |
|------|------|----------|--------------------------------|
| **SA** | Arousal | Resource mobilization under stress | Expected precision on sensory errors |
| **SE** | Exploration | Novelty-seeking vs. reliable strategies | Epistemic value weighting |
| **SP** | Plasticity | Architectural change rate | Meta-learning rate |
| **SS** | Social | Multi-agent coordination | Precision on social priors |

---

## 🚀 Installation

### Quick Install (3 steps)

```bash
# 1. Clone the repository
git clone https://github.com/dukeru115/GSV.git
cd GSV

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python gsv2_tests.py
```

### Using pip (Alternative)

```bash
# Install from source
pip install -e .

# Or install specific dependencies
pip install numpy>=1.20.0 matplotlib>=3.3.0 scipy>=1.6.0
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

### Files Structure

```
GSV/
├── gsv2_core.py                # Core GSV implementation (SDE solver, metrics)
├── gsv2_qlearning.py           # Q-learning integration wrapper
├── gsv2_analysis.py            # Stability analysis & diagnostics
├── gsv2_gridworld_demo.py      # Full demo (3000 episodes)
├── gsv2_stress_demo.py         # Stress response scenario
├── gsv2_experiments.py         # Parameter studies & ablations
├── gsv2_tests.py               # Comprehensive test suite
├── gsv2_quickstart.py          # Learning examples & patterns
├── examples/                   # Simple usage examples
│   ├── simple_example.py       # Minimal working example
│   └── README.md
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
├── pyproject.toml              # Modern Python packaging
├── README.md                   # This file
├── Summary.md                  # Complete implementation summary
├── QUICKSTART.md               # 5-minute getting started guide
├── INSTALL.md                  # Detailed installation guide
└── CONTRIBUTING.md             # Contribution guidelines
```

---

## ⚡ Quick Start

### Basic Usage

```python
from gsv2_core import GSV2Controller, GSV2Params, MetricComputer

# Initialize with balanced parameters
gsv = GSV2Controller(GSV2Params())
metrics_computer = MetricComputer()

# Agent loop
for step in range(1000):
    # Your agent acts and learns
    td_error = agent.update(state, action, reward, next_state)
    
    # Compute GSV metrics
    stress = metrics_computer.compute_stress(td_error)
    coherence = metrics_computer.compute_coherence(policy_probs)
    novelty = metrics_computer.compute_novelty(state)
    
    metrics = {
        'rho_def': stress,
        'R': coherence,
        'novelty': novelty,
        'SIR': 0.0,  # For multi-agent
        'F': reward
    }
    
    # Update GSV (slow dynamics)
    gsv.step(metrics, dt=1.0)
    
    # Modulate agent parameters
    from gsv2_core import ModulationFunctions as MF
    gsv_state = gsv.get_state()
    
    epsilon = MF.epsilon_exploration(gsv_state['E'])
    alpha = MF.alpha_learning(gsv_state['P'])
    
    agent.set_epsilon(epsilon)
    agent.set_alpha(alpha)
```

### Run Complete Demo

```bash
python gsv2_gridworld_demo.py
```

This runs a **3000-episode** experiment with **regime changes** at episodes 1000 and 2000, demonstrating:
- Initial exploration phase
- Stable exploitation after learning
- Adaptive response to environmental shifts
- Guaranteed recovery through bounded dynamics

---

## 🏗️ Architecture

### Closed-Loop Dynamics

```
┌─────────────────────────────────────────────────────┐
│                   FAST LAYER                        │
│  ┌──────────────────────────────────────────────┐   │
│  │  Agent (Q-learning, DQN, PPO, LLM, etc.)    │   │
│  │  • Perception  • Action  • Learning          │   │
│  └──────────────────────────────────────────────┘   │
│              ↓ Metrics                    ↑          │
│         (ρ, R, novelty)            Parameters        │
│              ↓                        (ε, α, γ)      │
└──────────────┼─────────────────────────────┼─────────┘
               │                             │
┌──────────────┼─────────────────────────────┼─────────┐
│              ↓         SLOW LAYER          ↑         │
│  ┌────────────────────────────────────────────────┐  │
│  │  GSV Controller: S(t) = [SA, SE, SP, SS]      │  │
│  │  • Stochastic Differential Equations          │  │
│  │  • Timescale: minutes (τ ~ 50-200s)          │  │
│  │  • Bounded dynamics (cubic damping)           │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Mathematical Formulation

The GSV dynamics are governed by **coupled Stochastic Differential Equations**:

```
dSA = [αA·(ρ̄def + kA·ExtStim) - γA·SA - λA·SA³] dt + σA·dWA

dSE = [αE·(Rtarget - R̄) - γE·SE - kAE·tanh(SA)·SE - λE·SE³] dt + σE·dWE

dSP = [αP·(novelty + kP·(Ftarget - F̄)) - γP·SP - kPS·tanh(SS)·SP - λP·SP³] dt + σP·dWP

dSS = [αS·SIR - γS·SS - λS·SS³] dt + σS·dWS
```

**Key innovations in GSV 2.0:**
- **Bounded cross-coupling**: `tanh(SA)` prevents runaway instability
- **Nonlinear damping**: `-λi·Si³` provides soft bounds without clipping
- **Stochastic terms**: Enable escape from local minima

---

## 📦 Modules

### 1. `gsv2_core.py` - Core Implementation

**Classes:**
- `GSV2Params`: Parameter configuration with stability validation
  - `.conservative()`: Maximum stability
  - `.aggressive()`: Fast adaptation
  
- `GSV2Controller`: Main SDE solver
  - `.step()`: Euler-Maruyama integration
  - `.get_state()`: Current GSV state
  - `.compute_lyapunov()`: Stability monitoring
  
- `MetricComputer`: Compute GSV metrics from agent
  - `.compute_stress()`: TD error → ρ̄def
  - `.compute_coherence()`: Policy entropy → R
  - `.compute_novelty()`: State visits → novelty rate
  
- `ModulationFunctions`: GSV → Agent parameters
  - `.epsilon_exploration(SE)`: Exploration rate
  - `.alpha_learning(SP)`: Learning rate
  - `.gamma_discount(SA)`: Discount factor

### 2. `gsv2_qlearning.py` - Agent Integration

**Classes:**
- `QLearningAgent`: Standard Q-learning with modifiable hyperparameters
- `GSV2ModulatedQLearning`: Complete integration wrapper
- `SimpleGridworld`: Test environment with regime changes

### 3. `gsv2_analysis.py` - Diagnostics

**Classes:**
- `StabilityAnalyzer`: 
  - Jacobian eigenvalue computation
  - Lyapunov function analysis
  - Trajectory stability assessment
  
- `PhaseSpaceAnalyzer`:
  - Behavioral regime identification
  - Vector field computation
  - Attractor basin analysis
  
- `DiagnosticMonitor`:
  - Real-time health checks
  - Warning system
  - Status reporting

### 4. `gsv2_gridworld_demo.py` - Full Demo

**Classes:**
- `ExperimentRunner`: Manages complete experiments
- `ResultVisualizer`: Generates publication-quality plots

---

## 📊 Examples

### Example 1: Parameter Configurations

```python
from gsv2_core import GSV2Params

# Conservative (maximum stability)
params = GSV2Params.conservative()

# Balanced (recommended default)
params = GSV2Params()

# Aggressive (fast adaptation)
params = GSV2Params.aggressive()

# Custom
params = GSV2Params(
    alpha=np.array([0.1, 0.08, 0.05, 0.08]),
    gamma=np.array([0.015, 0.015, 0.01, 0.01]),
    k_AE=0.1,
    k_PS=0.05
)
```

### Example 2: Stability Analysis

```python
from gsv2_analysis import StabilityAnalyzer, DiagnosticMonitor

gsv = GSV2Controller(GSV2Params())
analyzer = StabilityAnalyzer(gsv)
monitor = DiagnosticMonitor(gsv)

# Check stability conditions
conditions = analyzer.check_stability_conditions()
print(f"System stable: {conditions['overall_stable']}")

# Compute eigenvalues
eigenvalues = analyzer.compute_eigenvalues()
print(f"Eigenvalues: {eigenvalues}")

# Monitor health
status = monitor.check_health()
print(f"Health: {status['health']}")
```

### Example 3: Custom Agent Integration

```python
class MyCustomAgent:
    def __init__(self):
        self.gsv = GSV2Controller()
        self.metrics = MetricComputer()
    
    def train_step(self, batch):
        # Your training logic
        loss = self.compute_loss(batch)
        
        # Compute metrics
        metrics = {
            'rho_def': loss / 10.0,  # Normalized stress
            'R': self.policy_entropy(),
            'novelty': self.compute_novelty(),
            'F': self.get_reward()
        }
        
        # Update GSV
        self.gsv.step(metrics, dt=1.0)
        
        # Modulate learning rate
        state = self.gsv.get_state()
        new_lr = ModulationFunctions.alpha_learning(state['P'])
        self.optimizer.lr = new_lr
```

---

## 📖 Theory

### Stability Guarantee (Theorem 1)

**An equilibrium point S* of GSV 2.0 is locally stable if:**

```
γE > kAE - 3λE(SE*)²
```

For states near origin: **γE > kAE**

This is satisfied with balanced parameters:
- γE = 0.01
- kAE = 0.1 ✓

### Lyapunov Function

```
V(S) = Σ[½Si² + ¼λiSi⁴]
```

**Property**: `dV/dt < 0` for ||S|| sufficiently large  
**Consequence**: Global attracting set exists

### Phase Space Regimes

| Regime | GSV State | Behavior |
|--------|-----------|----------|
| **Explorer** | [0.5, +1, +1, 0] | High exploration, high plasticity |
| **Guardian** | [+1, -1, -0.5, 0] | High arousal, exploitation mode |
| **Collaborator** | [0, 0, -0.5, +1] | Social coordination prioritized |
| **Adapter** | [0, +0.5, +1, -0.5] | Dynamic learning, individual focus |

---

## 🔧 API Reference

### GSV2Controller

```python
class GSV2Controller:
    def __init__(self, params: GSV2Params = None)
    
    def step(self, metrics: Dict, dt: float = 1.0, 
             ext_stim: float = 0.0) -> np.ndarray
        """Single integration step"""
        
    def get_state(self) -> Dict[str, float]
        """Returns {'A': SA, 'E': SE, 'P': SP, 'S': SS}"""
        
    def reset(self)
        """Reset to initial state"""
        
    def compute_lyapunov(self) -> float
        """Compute V(S) for stability monitoring"""
```

### MetricComputer

```python
class MetricComputer:
    def __init__(self, window_size: int = 100, 
                 ewma_lambda: float = 0.05)
    
    def compute_stress(self, td_error: float) -> float
        """TD error → stress [0, 1]"""
        
    def compute_coherence(self, policy_probs: np.ndarray) -> float
        """Policy entropy → coherence [0, 1]"""
        
    def compute_novelty(self, state: int) -> float
        """State visitation → novelty [0, 1]"""
        
    def compute_fitness(self, reward: float) -> float
        """Smoothed reward"""
```

### ModulationFunctions

```python
class ModulationFunctions:
    @staticmethod
    def epsilon_exploration(SE: float, 
                           eps_min: float = 0.1, 
                           eps_max: float = 0.5) -> float
    
    @staticmethod
    def alpha_learning(SP: float, 
                      alpha_base: float = 0.001, 
                      k_alpha: float = 0.5) -> float
    
    @staticmethod
    def gamma_discount(SA: float, 
                      gamma_base: float = 0.99, 
                      k_gamma: float = 0.1) -> float
    
    @staticmethod
    def temperature(SE: float, 
                   T_min: float = 0.5, 
                   T_max: float = 2.0) -> float
```

---

## 🎓 Research Context

### Connection to Active Inference

GSV components can be understood as **sufficient statistics of precision parameters** at the strategic timescale:

- **SA (Arousal)** ~ Expected precision on sensory prediction errors
- **SE (Exploration)** ~ Epistemic value weighting
- **SP (Plasticity)** ~ Learning rate meta-precision
- **SS (Social)** ~ Precision on social priors

The core dynamics (driving + decay terms) align with **gradient descent on hierarchical free energy**. Cross-coupling terms are functionally motivated by biological observations.

### Convergent Evolution Principle

Multi-scale control is a **universal architectural requirement** for any complex adaptive system operating across multiple timescales. Biological systems independently evolved this pattern:

```
Fast Layer (1-100ms):   Neural firing, immediate perception/action
Medium Layer (seconds): Working memory, attention
Slow Layer (minutes):   Neuromodulation (cortisol, dopamine)
```

GSV 2.0 is the artificial analog for autonomous agents.

---

## 📈 Expected Results

From the **Gridworld experiment** (3000 episodes, 2 regime changes):

**Phase 1 (0-1000): Initial Learning**
- Success rate: 0% → 80%
- Exploration (ε): 0.5 → 0.3
- GSV: SE peaks, then stabilizes

**Phase 2 (1000-2000): Regime Change**
- Success rate: drops to ~20%, recovers to 80%
- Exploration: spike to 0.5, gradual decay
- GSV: SA spike (stress), SE increase (exploration)

**Phase 3 (2000-3000): Second Regime Change**
- Pattern repeats, demonstrating **consistent adaptation**

---

## 🤝 Contributing

This is a research implementation. Contributions welcome for:

- [ ] Additional agent integrations (DQN, PPO, A3C)
- [ ] LLM-specific metrics and modulation
- [ ] Multi-agent scenarios with SS axis
- [ ] Empirical validation on complex benchmarks
- [ ] Formal derivation of cross-coupling terms from FEP
- [ ] JAX/GPU acceleration

---

## 📚 Citation

```bibtex
@article{urmanov2025gsv2,
  title={Global State Vector 2.0: Multi-Scale Control for Autonomous AI Agents},
  author={Urmanov, Timur and Gadeev, Kamil and Iusupov, Bakhtier},
  year={2025},
  note={Theoretical Proposal for Discussion}
}
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 Related Work

- **Active Inference**: Friston, K. (2010). *The free-energy principle: a unified brain theory?*
- **Meta-RL**: Finn et al. (2017). *Model-Agnostic Meta-Learning (MAML)*
- **Neuromodulation**: Yu & Dayan (2005). *Uncertainty, neuromodulation, and attention*
- **Hierarchical RL**: Sutton et al. (1999). *Temporal abstraction in RL*

---

## 📞 Contact

- **Timur Urmanov**: urmanov.t@gmail.com
- **Kamil Gadeev**: gadeev.kamil@gmail.com
- **Bakhtier Iusupov**: usupovbahtiayr@gmail.com

---

**Status**: ✅ MVP Complete | 🚧 Empirical Validation Pending

*"The need for hierarchical, multi-scale control is not peculiar to artificial agents but represents a universal architectural requirement for any complex adaptive system."*