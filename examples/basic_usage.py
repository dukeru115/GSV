"""
Basic usage example of the GSV architecture.

This demonstrates how to set up a GSV agent with goals, strategies, and values
for a simple task environment.
"""

from gsv import GSVAgent, Goal, Strategy, ValueMetric
from gsv.goals import GoalModule
from gsv.strategies import StrategyModule
from gsv.values import ValueModule


def simple_evaluator(observation, action, goal, context):
    """Simple reward evaluator for demonstration."""
    # In a real scenario, this would evaluate actual task performance
    if observation is None:
        return 0.5
    
    # Simulate performance based on observation
    reward = observation.get("reward", 0.5) if isinstance(observation, dict) else 0.5
    return max(0.0, min(1.0, reward))


def custom_action_policy(observation, strategies, context):
    """Custom action selection policy."""
    # In a real scenario, this would map strategies to environment actions
    return {
        "action_type": "custom",
        "strategies": strategies,
        "observation": observation
    }


def main():
    # Create modules
    goal_module = GoalModule()
    strategy_module = StrategyModule()
    value_module = ValueModule()
    
    # Define goals
    goal1 = Goal(
        name="maximize_reward",
        description="Maximize cumulative reward over time",
        priority=1.0,
        success_threshold=0.9,
        time_horizon=500
    )
    goal_module.add_goal(goal1)
    
    goal2 = Goal(
        name="explore_environment",
        description="Explore different areas of the environment",
        priority=0.7,
        success_threshold=0.8,
        time_horizon=300
    )
    goal_module.add_goal(goal2)
    
    # Define strategies
    strategy1 = Strategy(
        name="exploit",
        description="Exploit known good actions",
        parameters={
            "exploration_rate": 0.1,
            "learning_rate": 0.01,
            "discount_factor": 0.95
        }
    )
    strategy_module.add_strategy(strategy1)
    
    strategy2 = Strategy(
        name="explore",
        description="Explore new actions",
        parameters={
            "exploration_rate": 0.5,
            "learning_rate": 0.05,
            "discount_factor": 0.9
        }
    )
    strategy_module.add_strategy(strategy2)
    
    strategy_module.set_active_strategies(["exploit"])
    strategy_module.set_action_policy(custom_action_policy)
    
    # Define value metrics
    reward_metric = ValueMetric(
        name="reward",
        description="Immediate reward from environment",
        weight=1.0,
        target_value=1.0
    )
    value_module.add_metric(reward_metric)
    value_module.add_evaluator("reward", simple_evaluator)
    
    efficiency_metric = ValueMetric(
        name="efficiency",
        description="Actions per unit reward",
        weight=0.5,
        target_value=0.8
    )
    value_module.add_metric(efficiency_metric)
    value_module.add_evaluator("efficiency", lambda obs, act, goal, ctx: 0.7)
    
    # Create GSV agent
    agent = GSVAgent(
        goal_module=goal_module,
        strategy_module=strategy_module,
        value_module=value_module,
        adaptation_rate=0.1
    )
    
    print("=" * 60)
    print("GSV Agent Demonstration")
    print("=" * 60)
    print()
    
    # Run simulation
    print("Running simulation for 20 steps...")
    print()
    
    for step in range(20):
        # Simulate observation
        observation = {
            "reward": 0.5 + (step % 5) * 0.1,  # Varying reward
            "state": step
        }
        
        # Agent takes a step
        action = agent.step(observation)
        
        # Print periodic updates
        if (step + 1) % 5 == 0:
            print(f"Step {step + 1}:")
            print(f"  Action: {action}")
            
            state = agent.get_state()
            print(f"  Current Goal: {state.current_goal}")
            print(f"  Active Strategies: {state.active_strategies}")
            print(f"  Performance Metrics: {state.performance_metrics}")
            print()
    
    # Get interpretable summary
    print("=" * 60)
    print("Final Agent State Summary")
    print("=" * 60)
    print()
    
    summary = agent.get_interpretable_summary()
    
    print(f"Current Goal: {summary['current_goal']}")
    print()
    
    print("Goal Details:")
    for goal_name, details in summary['goal_details']['goals'].items():
        print(f"  {goal_name}:")
        print(f"    Description: {details['description']}")
        print(f"    Priority: {details['priority']}")
        print(f"    Success Threshold: {details['success_threshold']}")
    print()
    
    print(f"Active Strategies: {summary['active_strategies']}")
    print()
    
    print("Strategy Details:")
    for strategy_name, details in summary['strategy_details']['strategies'].items():
        print(f"  {strategy_name}:")
        print(f"    Description: {details['description']}")
        print(f"    Avg Performance: {details['avg_performance']:.3f}")
        print(f"    Parameters: {details['parameters']}")
    print()
    
    print("Value Assessments:")
    value_assessments = summary['value_assessments']
    print(f"  Overall Performance: {value_assessments['overall_performance']:.3f}")
    for metric_name, assessment in value_assessments['metrics'].items():
        print(f"  {metric_name}:")
        print(f"    Current: {assessment['current_value']:.3f}")
        print(f"    Trend: {assessment['trend']}")
        if 'gap' in assessment:
            print(f"    Gap to Target: {assessment['gap']:.3f}")
    print()
    
    print(f"Total Cycles: {summary['cycle_count']}")
    print(f"Adaptation Rate: {summary['adaptation_rate']}")
    print()
    
    print("=" * 60)
    print("Key Features Demonstrated:")
    print("=" * 60)
    print("✓ Multi-timescale cognition (reactive, tactical, strategic)")
    print("✓ Interpretable goal-strategy-value separation")
    print("✓ Adaptive parameter adjustment based on performance")
    print("✓ Clear state inspection and monitoring")
    print("✓ Theoretical grounding in hierarchical control")


if __name__ == "__main__":
    main()
