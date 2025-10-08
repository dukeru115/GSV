"""
Goal Module: Long-term strategic planning and goal management.

This module handles the slowest timescale (minutes to days), maintaining
and adapting high-level objectives based on environmental dynamics and
performance history.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Goal:
    """Represents a high-level objective with success criteria."""
    name: str
    description: str
    priority: float = 1.0
    success_threshold: float = 0.8
    time_horizon: int = 1000  # Expected cycles to completion
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate_success(self, performance: float) -> bool:
        """Check if goal has been achieved."""
        return performance >= self.success_threshold


class GoalModule:
    """
    Manages long-term goals and strategic direction.
    
    The Goal Module operates on the strategic timescale, determining
    what the agent should ultimately achieve. Goals provide:
    - Clear direction for lower-level strategies
    - Success criteria for evaluation
    - Interpretable high-level objectives
    """
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.active_goal: Optional[str] = None
        self.goal_history: List[Dict[str, Any]] = []
    
    def add_goal(self, goal: Goal):
        """Add a new goal to the system."""
        self.goals[goal.name] = goal
        logger.info(f"Goal added: {goal.name}")
        
        # Set as active if no active goal
        if self.active_goal is None:
            self.active_goal = goal.name
    
    def set_active_goal(self, goal_name: str):
        """Set the currently active goal."""
        if goal_name not in self.goals:
            raise ValueError(f"Goal '{goal_name}' not found")
        
        old_goal = self.active_goal
        self.active_goal = goal_name
        
        self.goal_history.append({
            "timestamp": datetime.now(),
            "from_goal": old_goal,
            "to_goal": goal_name,
            "reason": "manual_switch"
        })
        
        logger.info(f"Active goal changed: {old_goal} -> {goal_name}")
    
    def evaluate_goals(
        self,
        current_performance: float,
        metrics: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate whether current goals are appropriate and adapt if needed.
        
        This is the core strategic adaptation mechanism, operating on the
        slowest timescale to ensure stability while allowing for long-term
        behavioral changes.
        """
        updates = {}
        
        if self.active_goal is None:
            return updates
        
        active = self.goals[self.active_goal]
        
        # Check if current goal is achieved
        if active.evaluate_success(current_performance):
            updates["goal_achieved"] = self.active_goal
            
            # Find next priority goal
            next_goal = self._find_next_goal(exclude=[self.active_goal])
            if next_goal:
                self.set_active_goal(next_goal)
                updates["new_goal"] = next_goal
        
        # Check if goal priority should be adjusted based on context
        priority_updates = self._adjust_priorities(metrics, context)
        if priority_updates:
            updates["priority_adjustments"] = priority_updates
        
        return updates
    
    def _find_next_goal(self, exclude: Optional[List[str]] = None) -> Optional[str]:
        """Find the highest priority goal that hasn't been excluded."""
        exclude = exclude or []
        
        available_goals = [
            (name, goal) for name, goal in self.goals.items()
            if name not in exclude
        ]
        
        if not available_goals:
            return None
        
        # Return goal with highest priority
        return max(available_goals, key=lambda x: x[1].priority)[0]
    
    def _adjust_priorities(
        self,
        metrics: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adjust goal priorities based on current performance and context."""
        adjustments = {}
        
        # This is where domain-specific priority adjustment logic would go
        # For now, we keep priorities stable unless explicitly changed
        
        return adjustments
    
    def get_active_goal(self) -> Optional[str]:
        """Get the name of the currently active goal."""
        return self.active_goal
    
    def get_goal_details(self) -> Dict[str, Any]:
        """Get detailed information about all goals."""
        return {
            "active_goal": self.active_goal,
            "goals": {
                name: {
                    "description": goal.description,
                    "priority": goal.priority,
                    "success_threshold": goal.success_threshold,
                    "time_horizon": goal.time_horizon,
                }
                for name, goal in self.goals.items()
            },
            "history": self.goal_history[-5:],  # Recent history
        }
    
    def remove_goal(self, goal_name: str):
        """Remove a goal from the system."""
        if goal_name in self.goals:
            del self.goals[goal_name]
            
            if self.active_goal == goal_name:
                self.active_goal = self._find_next_goal()
            
            logger.info(f"Goal removed: {goal_name}")
