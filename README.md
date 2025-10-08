# GSV: Goal-Strategy-Value Architecture

A multi-timescale cognitive architecture for autonomous AI agents that enables robust long-term operation through interpretable, adaptive, and theoretically-grounded decision-making.

## Overview

Autonomous AI agents operating in complex, dynamic environments face a fundamental architectural challenge: **fast, reactive cognition (milliseconds to seconds) is insufficient for genuine autonomy**, which requires **strategic behavioral adaptation over much longer timescales (minutes to days)**.

Current approaches, such as fixed hyperparameters or black-box meta-learning, lack the interpretability, adaptivity, and theoretical grounding necessary for robust long-term operation.

**GSV** solves this by implementing a hierarchical architecture with three coordinated modules operating across multiple timescales:

- **Goal Module** (Strategic, minutes-days): Maintains and adapts high-level objectives
- **Strategy Module** (Tactical, seconds-minutes): Selects and adjusts behavioral approaches
- **Value Module** (Continuous): Evaluates performance and guides adaptation

## Key Features

### 1. Multi-Timescale Cognition
- **Reactive Level** (milliseconds-seconds): Fast action selection
- **Tactical Level** (seconds-minutes): Strategy adaptation
- **Strategic Level** (minutes-days): Goal evaluation and switching

### 2. Interpretability
- Clear separation of goals, strategies, values
- Explicit parameter tracking with adaptation history
- Human-readable state summaries at any time

### 3. Adaptivity
- Continuous parameter adjustment based on performance feedback
- Strategy switching when approaches become ineffective
- Goal re-prioritization based on achievement and context

### 4. Theoretical Grounding
- Based on hierarchical control theory
- Principled separation of timescales prevents instability
- Performance-driven adaptation with clear feedback loops

## Installation

```bash
# Clone the repository
git clone https://github.com/dukeru115/GSV.git
cd GSV

# Install in development mode
pip install -e .
```

## Quick Start

```python
from gsv import GSVAgent, Goal, Strategy, ValueMetric
from gsv.goals import GoalModule
from gsv.strategies import StrategyModule
from gsv.values import ValueModule

# Create modules
goal_module = GoalModule()
strategy_module = StrategyModule()
value_module = ValueModule()

# Define a goal
goal = Goal(
    name="maximize_reward",
    description="Maximize cumulative reward over time",
    priority=1.0,
    success_threshold=0.9
)
goal_module.add_goal(goal)

# Define a strategy
strategy = Strategy(
    name="exploit",
    description="Exploit known good actions",
    parameters={
        "exploration_rate": 0.1,
        "learning_rate": 0.01
    }
)
strategy_module.add_strategy(strategy)
strategy_module.set_active_strategies(["exploit"])

# Define value metrics
metric = ValueMetric(
    name="reward",
    description="Immediate reward from environment",
    weight=1.0
)
value_module.add_metric(metric)
value_module.add_evaluator("reward", lambda obs, act, goal, ctx: obs.get("reward", 0.5))

# Create GSV agent
agent = GSVAgent(
    goal_module=goal_module,
    strategy_module=strategy_module,
    value_module=value_module,
    adaptation_rate=0.1
)

# Run agent
for step in range(100):
    observation = get_observation()  # Your environment
    action = agent.step(observation)
    execute_action(action)  # Your environment
    
    # Inspect agent state
    if step % 10 == 0:
        summary = agent.get_interpretable_summary()
        print(f"Goal: {summary['current_goal']}")
        print(f"Strategies: {summary['active_strategies']}")
        print(f"Performance: {summary['performance_metrics']}")
```

## Architecture

### Goal Module
The Goal Module operates on the **strategic timescale**, managing long-term objectives:

- Maintains a set of goals with priorities and success criteria
- Evaluates goal achievement based on performance metrics
- Switches between goals when objectives are met or priorities change
- Provides interpretable high-level direction

### Strategy Module
The Strategy Module operates on the **tactical timescale**, managing behavioral approaches:

- Maintains a repertoire of strategies with adaptive parameters
- Selects strategies based on current goals and past performance
- Adjusts strategy parameters through performance feedback
- Switches strategies when current approaches are ineffective

### Value Module
The Value Module provides continuous **performance evaluation**:

- Tracks multiple performance metrics with different weights
- Analyzes trends to detect improvement or degradation
- Provides feedback signals for adaptation at all timescales
- Enables interpretable assessment of agent behavior

### GSV Agent
The GSV Agent coordinates all modules across timescales:

```
Strategic (slow):  Goals ────────────────> Evaluate & Switch
                     ↓                            ↑
Tactical (medium): Strategies ─────────> Adapt Parameters
                     ↓                            ↑
Reactive (fast):   Actions ──────────────> Execute
                     ↓                            ↑
                   Environment ───────────> Performance
                                   (Value Module)
```

## Examples

See the `examples/` directory for detailed usage:

- `basic_usage.py`: Simple demonstration of all components
- More examples coming soon!

## Design Principles

1. **Separation of Concerns**: Goals, strategies, and values are explicitly separated
2. **Multiple Timescales**: Different cognitive processes operate at appropriate rates
3. **Performance-Driven**: All adaptation is based on measurable performance feedback
4. **Interpretability First**: Every decision and adaptation can be inspected and understood
5. **Minimal Assumptions**: Framework is general and can be specialized for any domain

## Advantages over Alternatives

| Approach | Interpretability | Adaptivity | Theoretical Grounding | Long-term Stability |
|----------|------------------|------------|----------------------|---------------------|
| Fixed Hyperparameters | ✓ | ✗ | ✓ | ✓ |
| Black-box Meta-learning | ✗ | ✓ | ✗ | ? |
| **GSV Architecture** | ✓ | ✓ | ✓ | ✓ |

## Use Cases

- **Autonomous Robots**: Long-term operation with changing objectives
- **Game AI**: Adaptive opponents that learn and change tactics
- **Resource Management**: Dynamic optimization across timescales
- **Continuous Learning**: Agents that improve over extended periods
- **Multi-objective Systems**: Balancing competing goals adaptively

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Citation

If you use GSV in your research, please cite:

```bibtex
@software{gsv2024,
  title={GSV: Goal-Strategy-Value Architecture for Autonomous AI Agents},
  author={GSV Contributors},
  year={2024},
  url={https://github.com/dukeru115/GSV}
}
```

## References

This architecture is inspired by:
- Hierarchical Reinforcement Learning
- Control Theory and Multi-rate Control Systems
- Cognitive Architectures (ACT-R, SOAR)
- Meta-learning and Adaptive Systems