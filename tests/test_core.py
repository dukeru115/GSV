"""Tests for core GSV agent functionality."""

import pytest
from gsv.core import GSVAgent, TimescaleLevel, CognitiveState
from gsv.goals import GoalModule, Goal
from gsv.strategies import StrategyModule, Strategy
from gsv.values import ValueModule, ValueMetric


class TestGSVAgent:
    """Test suite for GSVAgent class."""
    
    def test_agent_initialization(self):
        """Test that agent can be initialized with default modules."""
        agent = GSVAgent()
        assert agent is not None
        assert agent.goal_module is not None
        assert agent.strategy_module is not None
        assert agent.value_module is not None
        assert agent.cycle_count == 0
    
    def test_agent_with_custom_modules(self):
        """Test agent initialization with custom modules."""
        goal_module = GoalModule()
        strategy_module = StrategyModule()
        value_module = ValueModule()
        
        agent = GSVAgent(
            goal_module=goal_module,
            strategy_module=strategy_module,
            value_module=value_module
        )
        
        assert agent.goal_module is goal_module
        assert agent.strategy_module is strategy_module
        assert agent.value_module is value_module
    
    def test_agent_step(self):
        """Test that agent can execute a step."""
        agent = GSVAgent()
        
        # Add minimal setup
        goal = Goal(name="test_goal", description="Test goal")
        agent.goal_module.add_goal(goal)
        
        strategy = Strategy(name="test_strategy", description="Test strategy")
        agent.strategy_module.add_strategy(strategy)
        agent.strategy_module.set_active_strategies(["test_strategy"])
        
        metric = ValueMetric(name="test_metric", description="Test metric")
        agent.value_module.add_metric(metric)
        agent.value_module.add_evaluator(
            "test_metric",
            lambda obs, act, goal, ctx: 0.5
        )
        
        observation = {"state": 0}
        action = agent.step(observation)
        
        assert action is not None
        assert agent.cycle_count == 1
    
    def test_multi_timescale_updates(self):
        """Test that different timescales update at appropriate rates."""
        agent = GSVAgent()
        
        # Setup
        goal = Goal(name="test_goal", description="Test goal")
        agent.goal_module.add_goal(goal)
        
        strategy = Strategy(name="test_strategy", description="Test strategy")
        agent.strategy_module.add_strategy(strategy)
        agent.strategy_module.set_active_strategies(["test_strategy"])
        
        # Run multiple cycles
        for _ in range(100):
            agent.step({"state": 0})
        
        # Verify updates happened at different rates
        assert agent.cycle_count == 100
        
        # Strategic updates should be less frequent than tactical
        strategic_updates = sum(
            1 for entry in agent.state.adaptation_history
            if entry.get("timescale") == TimescaleLevel.STRATEGIC
        )
        tactical_updates = sum(
            1 for entry in agent.state.adaptation_history
            if entry.get("timescale") == TimescaleLevel.TACTICAL
        )
        
        # Strategic should update less frequently
        assert strategic_updates <= tactical_updates or strategic_updates == 0
    
    def test_get_state(self):
        """Test retrieving agent cognitive state."""
        agent = GSVAgent()
        state = agent.get_state()
        
        assert isinstance(state, CognitiveState)
        assert state.current_goal is None  # No goals set yet
        assert isinstance(state.active_strategies, list)
        assert isinstance(state.performance_metrics, dict)
    
    def test_get_interpretable_summary(self):
        """Test getting interpretable summary of agent state."""
        agent = GSVAgent()
        
        # Add some structure
        goal = Goal(name="test_goal", description="Test goal")
        agent.goal_module.add_goal(goal)
        
        summary = agent.get_interpretable_summary()
        
        assert "current_goal" in summary
        assert "goal_details" in summary
        assert "active_strategies" in summary
        assert "strategy_details" in summary
        assert "performance_metrics" in summary
        assert "value_assessments" in summary
        assert "cycle_count" in summary
        assert "adaptation_rate" in summary
    
    def test_agent_reset(self):
        """Test that agent state can be reset."""
        agent = GSVAgent()
        
        # Run some steps
        for _ in range(10):
            agent.step({"state": 0})
        
        assert agent.cycle_count == 10
        
        # Reset
        agent.reset()
        
        assert agent.cycle_count == 0
        assert agent.state.current_goal is None
    
    def test_adaptation_rate(self):
        """Test that adaptation rate is properly set."""
        agent = GSVAgent(adaptation_rate=0.2)
        assert agent.adaptation_rate == 0.2
    
    def test_custom_timescale_ratios(self):
        """Test custom timescale update ratios."""
        custom_ratios = {
            TimescaleLevel.REACTIVE: 1.0,
            TimescaleLevel.TACTICAL: 0.2,
            TimescaleLevel.STRATEGIC: 0.05,
        }
        
        agent = GSVAgent(timescale_ratios=custom_ratios)
        
        assert agent.timescale_ratios == custom_ratios
