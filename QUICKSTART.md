# GSV 2.0 Quick Start Guide

Get started with GSV 2.0 in 5 minutes!

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/dukeru115/GSV.git
cd GSV

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation (30 seconds)

```bash
# Run a simple example
python examples/simple_example.py
```

**Expected output:** GSV state evolution over 100 steps ✓

## Run Your First Demo (5 minutes)

```bash
# Run the full gridworld demonstration
python gsv2_gridworld_demo.py
```

This will:
- Train an agent for 3000 episodes
- Demonstrate adaptation to 2 regime changes
- Generate visualization (`gsv2_gridworld_results.png`)

## Basic Usage Pattern

Here's the minimal code to integrate GSV with your agent:

```python
from gsv2_core import GSV2Controller, MetricComputer, ModulationFunctions

# Initialize
gsv = GSV2Controller()
metrics = MetricComputer()

# Training loop
for episode in range(1000):
    for step in episode:
        # Your agent acts and learns
        td_error = agent.update(...)
        
        # Compute metrics
        stress = metrics.compute_stress(td_error)
        coherence = metrics.compute_coherence(policy_probs)
        
        # Update GSV
        gsv.step({'rho_def': stress, 'R': coherence, ...})
        
        # Modulate parameters
        state = gsv.get_state()
        agent.epsilon = ModulationFunctions.epsilon_exploration(state['E'])
```

## What GSV Does

GSV provides **strategic adaptation** through multi-scale control:

1. **Monitors** your agent's learning state (stress, coherence, novelty)
2. **Updates** a low-dimensional control state [Arousal, Exploration, Plasticity, Social]
3. **Modulates** agent parameters (ε, α, γ) based on accumulated experience

**Result:** Your agent automatically adapts exploration, learning rate, and planning horizon based on its performance.

## Key Components

### 1. GSV Controller
```python
gsv = GSV2Controller()          # Initialize with default parameters
state = gsv.step(metrics)       # Update (once per episode or batch)
current = gsv.get_state()       # Get [A, E, P, S] state
```

### 2. Metric Computer
```python
metrics = MetricComputer()
stress = metrics.compute_stress(td_error)
coherence = metrics.compute_coherence(policy_distribution)
novelty = metrics.compute_novelty(state_id)
```

### 3. Modulation Functions
```python
epsilon = ModulationFunctions.epsilon_exploration(SE)  # SE → ε ∈ [0.1, 0.5]
alpha = ModulationFunctions.alpha_learning(SP)         # SP → α (exponential)
gamma = ModulationFunctions.gamma_discount(SA)         # SA → γ ∈ [0.9, 0.99]
```

## Parameter Configurations

### Conservative (Maximum Stability)
```python
params = GSV2Params.conservative()
gsv = GSV2Controller(params)
```
**Use when:** You need guaranteed stability, slow careful adaptation

### Balanced (Default)
```python
params = GSV2Params()  # or GSV2Controller()
gsv = GSV2Controller(params)
```
**Use when:** Most cases, good balance of adaptation and stability

### Aggressive (Fast Adaptation)
```python
params = GSV2Params.aggressive()
gsv = GSV2Controller(params)
```
**Use when:** You need quick responses to environment changes

## Next Steps

### Learn More
- **Full documentation**: See `README.md`
- **More examples**: Check `gsv2_quickstart.py`
- **Theory**: Read `Summary.md`

### Run Tests
```bash
python gsv2_tests.py
```

### Try Different Scenarios
```bash
python gsv2_stress_demo.py      # Stress response scenario
python gsv2_experiments.py      # Parameter comparison studies
```

### Integration Guide

To integrate GSV with your existing agent:

1. **Identify metrics**: What indicates stress/coherence in your domain?
2. **Choose update frequency**: Once per episode? Once per batch?
3. **Select parameters**: Start with conservative, tune as needed
4. **Monitor stability**: Use `DiagnosticMonitor` from `gsv2_analysis.py`

## Common Patterns

### Pattern 1: Episode-Based Updates
```python
for episode in episodes:
    episode_td_errors = []
    for step in episode:
        td_error = agent.step(...)
        episode_td_errors.append(td_error)
    
    # Update GSV once per episode
    avg_stress = np.mean([abs(e) for e in episode_td_errors])
    gsv.step({'rho_def': avg_stress, ...})
```

### Pattern 2: Batch-Based Updates
```python
for batch in batches:
    loss = train_on_batch(batch)
    
    # Update GSV once per batch
    gsv.step({'rho_def': loss / loss_scale, ...})
    
    # Apply to next batch
    state = gsv.get_state()
    learning_rate = ModulationFunctions.alpha_learning(state['P'])
```

### Pattern 3: Continuous Updates
```python
for step in steps:
    td_error = agent.step(...)
    
    # Update GSV every step (with small dt)
    gsv.step({'rho_def': abs(td_error), ...}, dt=0.1)
```

## Troubleshooting

### Issue: State diverges (||S|| > 5)
**Solution:** Use conservative parameters or decrease alpha

### Issue: No adaptation visible
**Solution:** Increase alpha or decrease gamma

### Issue: Stability warnings
**Solution:** Ensure γE > kAE in your parameters

## Support

- **Examples**: `examples/` directory and `gsv2_quickstart.py`
- **Documentation**: `README.md` and `Summary.md`
- **Issues**: https://github.com/dukeru115/GSV/issues
- **Email**: urmanov.t@gmail.com

## Citation

```bibtex
@article{urmanov2025gsv2,
  title={Global State Vector 2.0: Multi-Scale Control for Autonomous AI Agents},
  author={Urmanov, Timur and Gadeev, Kamil and Iusupov, Bakhtier},
  year={2025}
}
```

---

**Ready to build adaptive agents? Let's go!** 🚀
