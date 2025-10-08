"""
Core GSV Agent implementation with multi-timescale cognitive architecture.

This module implements the main GSVAgent class that coordinates goal-driven behavior
across multiple timescales: reactive (milliseconds-seconds), tactical (seconds-minutes),
and strategic (minutes-days).
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TimescaleLevel(Enum):
    """Cognitive timescale levels for multi-timescale decision making."""
    REACTIVE = "reactive"  # Milliseconds to seconds
    TACTICAL = "tactical"  # Seconds to minutes
    STRATEGIC = "strategic"  # Minutes to days


@dataclass
class CognitiveState:
    """Current state of the agent's cognitive system."""
    current_goal: Optional[str] = None
    active_strategies: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


class GSVAgent:
    """
    Goal-Strategy-Value Agent with multi-timescale cognition.
    
    The GSVAgent addresses the fundamental architectural challenge of combining:
    - Fast reactive cognition (milliseconds to seconds)
    - Strategic behavioral adaptation (minutes to days)
    
    This is achieved through three coordinated modules:
    - Goal Module: Defines and maintains long-term objectives
    - Strategy Module: Selects and adapts tactical approaches
    - Value Module: Evaluates performance and guides adaptation
    
    Unlike fixed hyperparameters or black-box meta-learning, GSV provides:
    - Interpretability: Clear separation of goals, strategies, and values
    - Adaptivity: Continuous parameter adjustment based on performance
    - Theoretical grounding: Principled multi-timescale architecture
    """
    
    def __init__(
        self,
        goal_module=None,
        strategy_module=None,
        value_module=None,
        adaptation_rate: float = 0.1,
        timescale_ratios: Optional[Dict[TimescaleLevel, float]] = None
    ):
        """
        Initialize the GSV Agent.
        
        Args:
            goal_module: Module for managing goals (default: GoalModule)
            strategy_module: Module for managing strategies (default: StrategyModule)
            value_module: Module for evaluating performance (default: ValueModule)
            adaptation_rate: Rate at which parameters adapt (0.0 to 1.0)
            timescale_ratios: Custom timescale update ratios
        """
        from .goals import GoalModule
        from .strategies import StrategyModule
        from .values import ValueModule
        
        self.goal_module = goal_module or GoalModule()
        self.strategy_module = strategy_module or StrategyModule()
        self.value_module = value_module or ValueModule()
        
        self.adaptation_rate = adaptation_rate
        self.timescale_ratios = timescale_ratios or {
            TimescaleLevel.REACTIVE: 1.0,     # Update every cycle
            TimescaleLevel.TACTICAL: 0.1,     # Update every 10 cycles
            TimescaleLevel.STRATEGIC: 0.01,   # Update every 100 cycles
        }
        
        self.state = CognitiveState()
        self.cycle_count = 0
        
        logger.info("GSVAgent initialized with adaptive multi-timescale cognition")
    
    def step(self, observation: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute one cognitive cycle across all timescales.
        
        Args:
            observation: Current environmental observation
            context: Additional contextual information
            
        Returns:
            Action to take in the environment
        """
        self.cycle_count += 1
        context = context or {}
        
        # Update strategic level (goals) - slowest timescale
        if self._should_update(TimescaleLevel.STRATEGIC):
            self._update_goals(observation, context)
        
        # Update tactical level (strategies) - medium timescale
        if self._should_update(TimescaleLevel.TACTICAL):
            self._update_strategies(observation, context)
        
        # Reactive level (action selection) - fastest timescale
        action = self._select_action(observation, context)
        
        # Evaluate and adapt based on performance
        self._evaluate_and_adapt(observation, action, context)
        
        return action
    
    def _should_update(self, timescale: TimescaleLevel) -> bool:
        """Determine if a timescale level should be updated this cycle."""
        ratio = self.timescale_ratios[timescale]
        return (self.cycle_count % max(1, int(1.0 / ratio))) == 0
    
    def _update_goals(self, observation: Any, context: Dict[str, Any]):
        """Update long-term goals based on performance and environmental changes."""
        current_performance = self.value_module.get_overall_performance()
        
        # Goal module analyzes whether current goals are still appropriate
        goal_updates = self.goal_module.evaluate_goals(
            current_performance,
            self.state.performance_metrics,
            context
        )
        
        if goal_updates:
            self.state.current_goal = self.goal_module.get_active_goal()
            self.state.adaptation_history.append({
                "timestamp": datetime.now(),
                "timescale": TimescaleLevel.STRATEGIC,
                "updates": goal_updates
            })
            logger.info(f"Goals updated: {goal_updates}")
    
    def _update_strategies(self, observation: Any, context: Dict[str, Any]):
        """Update tactical strategies based on current goal and performance."""
        current_goal = self.goal_module.get_active_goal()
        performance = self.value_module.get_strategy_performance()
        
        # Strategy module adapts tactical approaches
        strategy_updates = self.strategy_module.adapt_strategies(
            current_goal,
            performance,
            self.adaptation_rate,
            context
        )
        
        if strategy_updates:
            self.state.active_strategies = self.strategy_module.get_active_strategies()
            self.state.adaptation_history.append({
                "timestamp": datetime.now(),
                "timescale": TimescaleLevel.TACTICAL,
                "updates": strategy_updates
            })
            logger.info(f"Strategies updated: {strategy_updates}")
    
    def _select_action(self, observation: Any, context: Dict[str, Any]) -> Any:
        """Select reactive action based on current strategies."""
        active_strategies = self.strategy_module.get_active_strategies()
        
        # Fast reactive action selection
        action = self.strategy_module.select_action(
            observation,
            active_strategies,
            context
        )
        
        return action
    
    def _evaluate_and_adapt(self, observation: Any, action: Any, context: Dict[str, Any]):
        """Evaluate performance and trigger adaptation if needed."""
        # Value module tracks performance metrics
        metrics = self.value_module.evaluate(
            observation,
            action,
            self.state.current_goal,
            context
        )
        
        self.state.performance_metrics.update(metrics)
        self.state.last_update = datetime.now()
    
    def get_state(self) -> CognitiveState:
        """Get current cognitive state for inspection and debugging."""
        return self.state
    
    def get_interpretable_summary(self) -> Dict[str, Any]:
        """
        Get human-readable summary of agent's cognitive state.
        
        This provides interpretability by exposing:
        - Current goals and their rationale
        - Active strategies and their performance
        - Value metrics and adaptation history
        """
        return {
            "current_goal": self.state.current_goal,
            "goal_details": self.goal_module.get_goal_details(),
            "active_strategies": self.state.active_strategies,
            "strategy_details": self.strategy_module.get_strategy_details(),
            "performance_metrics": self.state.performance_metrics,
            "value_assessments": self.value_module.get_value_assessments(),
            "recent_adaptations": self.state.adaptation_history[-10:],
            "cycle_count": self.cycle_count,
            "adaptation_rate": self.adaptation_rate,
        }
    
    def reset(self):
        """Reset agent state while preserving learned adaptations."""
        self.state = CognitiveState()
        self.cycle_count = 0
        logger.info("Agent state reset")
