"""
Strategy Module: Tactical decision-making and behavioral adaptation.

This module handles the medium timescale (seconds to minutes), selecting
and adapting tactical approaches to achieve current goals.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class Strategy:
    """Represents a tactical approach with adaptive parameters."""
    name: str
    description: str
    parameters: Dict[str, float] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    activation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_average_performance(self) -> float:
        """Calculate average performance of this strategy."""
        if not self.performance_history:
            return 0.5  # Default neutral performance
        return sum(self.performance_history) / len(self.performance_history)
    
    def update_performance(self, performance: float):
        """Record new performance measurement."""
        self.performance_history.append(performance)
        # Keep history bounded
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)


class StrategyModule:
    """
    Manages tactical strategies and behavioral adaptation.
    
    The Strategy Module operates on the tactical timescale, determining
    how to pursue current goals. Strategies provide:
    - Flexible behavioral policies with adaptive parameters
    - Performance-based selection and adjustment
    - Interpretable tactical decisions
    """
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: List[str] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        self.action_policy: Optional[Callable] = None
    
    def add_strategy(self, strategy: Strategy):
        """Add a new strategy to the repertoire."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Strategy added: {strategy.name}")
    
    def set_action_policy(self, policy: Callable):
        """
        Set the action selection policy.
        
        Args:
            policy: Function that takes (observation, strategies, context) -> action
        """
        self.action_policy = policy
    
    def adapt_strategies(
        self,
        current_goal: Optional[str],
        performance: Dict[str, float],
        adaptation_rate: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt strategies based on current goal and performance.
        
        This implements the tactical adaptation mechanism:
        - Adjust strategy parameters based on performance feedback
        - Switch between strategies if current approach is ineffective
        - Maintain interpretable parameter changes
        """
        updates = {}
        
        if not self.strategies:
            return updates
        
        # Update performance records for active strategies
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies and strategy_name in performance:
                strategy = self.strategies[strategy_name]
                strategy.update_performance(performance[strategy_name])
        
        # Adapt parameters for active strategies
        parameter_updates = self._adapt_parameters(adaptation_rate, performance)
        if parameter_updates:
            updates["parameter_adjustments"] = parameter_updates
        
        # Consider switching strategies if performance is poor
        strategy_switches = self._evaluate_strategy_switches(performance, context)
        if strategy_switches:
            updates["strategy_switches"] = strategy_switches
        
        if updates:
            self.adaptation_history.append({
                "timestamp": datetime.now(),
                "goal": current_goal,
                "updates": updates
            })
        
        return updates
    
    def _adapt_parameters(
        self,
        adaptation_rate: float,
        performance: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Adapt strategy parameters based on performance feedback.
        
        Uses a simple gradient-based approach with random exploration
        to adjust parameters in the direction of better performance.
        """
        adjustments = {}
        
        for strategy_name in self.active_strategies:
            if strategy_name not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_name]
            current_perf = performance.get(strategy_name, 0.5)
            avg_perf = strategy.get_average_performance()
            
            # If current performance is better than average, continue in same direction
            # If worse, try adjusting parameters
            if current_perf < avg_perf - 0.1:  # Significant underperformance
                param_adjustments = {}
                
                for param_name, param_value in strategy.parameters.items():
                    # Add small random adjustment to explore parameter space
                    adjustment = random.gauss(0, adaptation_rate * 0.1)
                    new_value = max(0.0, min(1.0, param_value + adjustment))
                    
                    if abs(new_value - param_value) > 0.01:
                        strategy.parameters[param_name] = new_value
                        param_adjustments[param_name] = {
                            "old": param_value,
                            "new": new_value
                        }
                
                if param_adjustments:
                    adjustments[strategy_name] = param_adjustments
        
        return adjustments
    
    def _evaluate_strategy_switches(
        self,
        performance: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether to switch active strategies."""
        switches = {}
        
        # If active strategies are performing poorly, consider alternatives
        for strategy_name in list(self.active_strategies):
            if strategy_name not in self.strategies:
                continue
            
            strategy = self.strategies[strategy_name]
            current_perf = performance.get(strategy_name, 0.5)
            
            # If consistently underperforming, look for alternatives
            if (len(strategy.performance_history) >= 5 and
                strategy.get_average_performance() < 0.3):
                
                # Find better performing strategy
                alternatives = [
                    (name, s) for name, s in self.strategies.items()
                    if name not in self.active_strategies and
                    s.get_average_performance() > current_perf
                ]
                
                if alternatives:
                    best_alternative = max(alternatives, key=lambda x: x[1].get_average_performance())
                    self.active_strategies.remove(strategy_name)
                    self.active_strategies.append(best_alternative[0])
                    
                    switches[strategy_name] = {
                        "replaced_with": best_alternative[0],
                        "reason": "poor_performance",
                        "old_performance": current_perf,
                        "new_performance": best_alternative[1].get_average_performance()
                    }
        
        return switches
    
    def select_action(
        self,
        observation: Any,
        active_strategies: List[str],
        context: Dict[str, Any]
    ) -> Any:
        """
        Select action based on active strategies (reactive timescale).
        
        This is the fastest timescale, where tactical strategies are
        executed to produce concrete actions.
        """
        if self.action_policy:
            # Use custom policy if provided
            return self.action_policy(observation, active_strategies, context)
        
        # Default action selection: use first active strategy
        if active_strategies and active_strategies[0] in self.strategies:
            strategy = self.strategies[active_strategies[0]]
            strategy.activation_count += 1
            
            # Default action is a dictionary of strategy parameters
            # In real use, this would be mapped to environment-specific actions
            return {
                "strategy": strategy.name,
                "parameters": strategy.parameters.copy()
            }
        
        # No active strategy - return null action
        return {"strategy": "none", "parameters": {}}
    
    def get_active_strategies(self) -> List[str]:
        """Get list of currently active strategies."""
        return self.active_strategies.copy()
    
    def set_active_strategies(self, strategy_names: List[str]):
        """Set which strategies are currently active."""
        # Validate all strategy names exist
        for name in strategy_names:
            if name not in self.strategies:
                raise ValueError(f"Strategy '{name}' not found")
        
        self.active_strategies = strategy_names.copy()
        logger.info(f"Active strategies set to: {strategy_names}")
    
    def get_strategy_details(self) -> Dict[str, Any]:
        """Get detailed information about all strategies."""
        return {
            "active": self.active_strategies,
            "strategies": {
                name: {
                    "description": strategy.description,
                    "parameters": strategy.parameters,
                    "avg_performance": strategy.get_average_performance(),
                    "activation_count": strategy.activation_count,
                }
                for name, strategy in self.strategies.items()
            },
            "recent_adaptations": self.adaptation_history[-5:],
        }
