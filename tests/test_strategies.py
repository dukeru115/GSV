"""Tests for strategy module."""

import pytest
from gsv.strategies import StrategyModule, Strategy


class TestStrategy:
    """Test suite for Strategy class."""
    
    def test_strategy_creation(self):
        """Test basic strategy creation."""
        strategy = Strategy(
            name="test_strategy",
            description="A test strategy",
            parameters={"param1": 0.5}
        )
        
        assert strategy.name == "test_strategy"
        assert strategy.description == "A test strategy"
        assert strategy.parameters["param1"] == 0.5
        assert strategy.activation_count == 0
    
    def test_strategy_performance_tracking(self):
        """Test strategy performance tracking."""
        strategy = Strategy(name="test", description="Test")
        
        assert strategy.get_average_performance() == 0.5  # Default
        
        strategy.update_performance(0.8)
        strategy.update_performance(0.6)
        
        assert strategy.get_average_performance() == 0.7
    
    def test_strategy_performance_history_limit(self):
        """Test that performance history is bounded."""
        strategy = Strategy(name="test", description="Test")
        
        # Add many measurements
        for i in range(150):
            strategy.update_performance(0.5)
        
        # History should be bounded to 100
        assert len(strategy.performance_history) == 100


class TestStrategyModule:
    """Test suite for StrategyModule class."""
    
    def test_module_initialization(self):
        """Test strategy module can be initialized."""
        module = StrategyModule()
        
        assert module is not None
        assert len(module.strategies) == 0
        assert len(module.active_strategies) == 0
    
    def test_add_strategy(self):
        """Test adding strategies to the module."""
        module = StrategyModule()
        strategy = Strategy(name="strategy1", description="First")
        
        module.add_strategy(strategy)
        
        assert "strategy1" in module.strategies
    
    def test_add_multiple_strategies(self):
        """Test adding multiple strategies."""
        module = StrategyModule()
        
        strategy1 = Strategy(name="strategy1", description="First")
        strategy2 = Strategy(name="strategy2", description="Second")
        
        module.add_strategy(strategy1)
        module.add_strategy(strategy2)
        
        assert len(module.strategies) == 2
    
    def test_set_active_strategies(self):
        """Test setting active strategies."""
        module = StrategyModule()
        
        strategy1 = Strategy(name="strategy1", description="First")
        strategy2 = Strategy(name="strategy2", description="Second")
        
        module.add_strategy(strategy1)
        module.add_strategy(strategy2)
        
        module.set_active_strategies(["strategy1", "strategy2"])
        
        assert len(module.active_strategies) == 2
        assert "strategy1" in module.active_strategies
    
    def test_set_active_strategies_invalid(self):
        """Test setting non-existent strategy raises error."""
        module = StrategyModule()
        
        with pytest.raises(ValueError):
            module.set_active_strategies(["nonexistent"])
    
    def test_adapt_strategies_no_strategies(self):
        """Test adapting when no strategies exist."""
        module = StrategyModule()
        
        updates = module.adapt_strategies("goal1", {}, 0.1, {})
        
        assert len(updates) == 0
    
    def test_adapt_strategies_performance_update(self):
        """Test that strategies record performance updates."""
        module = StrategyModule()
        
        strategy = Strategy(name="strategy1", description="Test")
        module.add_strategy(strategy)
        module.set_active_strategies(["strategy1"])
        
        performance = {"strategy1": 0.8}
        module.adapt_strategies("goal1", performance, 0.1, {})
        
        assert len(strategy.performance_history) > 0
        assert strategy.performance_history[-1] == 0.8
    
    def test_select_action_default(self):
        """Test default action selection."""
        module = StrategyModule()
        
        strategy = Strategy(
            name="strategy1",
            description="Test",
            parameters={"param1": 0.5}
        )
        module.add_strategy(strategy)
        module.set_active_strategies(["strategy1"])
        
        action = module.select_action({}, ["strategy1"], {})
        
        assert action["strategy"] == "strategy1"
        assert "parameters" in action
        assert strategy.activation_count == 1
    
    def test_select_action_custom_policy(self):
        """Test custom action selection policy."""
        module = StrategyModule()
        
        def custom_policy(obs, strategies, context):
            return {"custom": True, "strategies": strategies}
        
        module.set_action_policy(custom_policy)
        
        action = module.select_action({}, ["strategy1"], {})
        
        assert action["custom"] is True
        assert action["strategies"] == ["strategy1"]
    
    def test_select_action_no_strategies(self):
        """Test action selection with no active strategies."""
        module = StrategyModule()
        
        action = module.select_action({}, [], {})
        
        assert action["strategy"] == "none"
    
    def test_get_active_strategies(self):
        """Test retrieving active strategies."""
        module = StrategyModule()
        
        strategy = Strategy(name="strategy1", description="Test")
        module.add_strategy(strategy)
        module.set_active_strategies(["strategy1"])
        
        active = module.get_active_strategies()
        
        assert len(active) == 1
        assert "strategy1" in active
    
    def test_get_strategy_details(self):
        """Test retrieving strategy details."""
        module = StrategyModule()
        
        strategy = Strategy(
            name="strategy1",
            description="Test strategy",
            parameters={"param1": 0.5}
        )
        module.add_strategy(strategy)
        module.set_active_strategies(["strategy1"])
        
        details = module.get_strategy_details()
        
        assert "active" in details
        assert "strategies" in details
        assert "strategy1" in details["strategies"]
        assert details["strategies"]["strategy1"]["description"] == "Test strategy"
