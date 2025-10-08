"""
Value Module: Performance evaluation and adaptation guidance.

This module evaluates agent performance across multiple dimensions and
provides feedback to guide goal and strategy adaptation.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValueMetric:
    """Represents a performance metric with tracking history."""
    name: str
    description: str
    weight: float = 1.0
    history: List[float] = field(default_factory=list)
    target_value: Optional[float] = None
    
    def add_measurement(self, value: float):
        """Add a new measurement to the history."""
        self.history.append(value)
        # Keep history bounded
        if len(self.history) > 1000:
            self.history.pop(0)
    
    def get_recent_average(self, window: int = 10) -> float:
        """Get average of recent measurements."""
        if not self.history:
            return 0.0
        recent = self.history[-window:]
        return sum(recent) / len(recent)
    
    def get_trend(self, window: int = 10) -> str:
        """Determine if metric is improving, declining, or stable."""
        if len(self.history) < window * 2:
            return "insufficient_data"
        
        older_avg = sum(self.history[-window*2:-window]) / window
        recent_avg = self.get_recent_average(window)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"


class ValueModule:
    """
    Evaluates agent performance and guides adaptation.
    
    The Value Module provides the evaluative function that determines
    how well the agent is performing. It supports:
    - Multiple performance metrics with different weights
    - Trend analysis for adaptation decisions
    - Interpretable performance assessments
    """
    
    def __init__(self):
        self.metrics: Dict[str, ValueMetric] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self.evaluators: Dict[str, Callable] = {}
    
    def add_metric(self, metric: ValueMetric):
        """Add a performance metric to track."""
        self.metrics[metric.name] = metric
        logger.info(f"Value metric added: {metric.name}")
    
    def add_evaluator(self, metric_name: str, evaluator: Callable):
        """
        Add an evaluator function for a metric.
        
        Args:
            metric_name: Name of the metric
            evaluator: Function that takes (observation, action, goal, context) -> float
        """
        self.evaluators[metric_name] = evaluator
    
    def evaluate(
        self,
        observation: Any,
        action: Any,
        current_goal: Optional[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate current performance across all metrics.
        
        This provides the feedback signal that drives adaptation at
        both strategic and tactical timescales.
        """
        measurements = {}
        
        # Evaluate each metric using its evaluator
        for metric_name, evaluator in self.evaluators.items():
            try:
                value = evaluator(observation, action, current_goal, context)
                measurements[metric_name] = value
                
                if metric_name in self.metrics:
                    self.metrics[metric_name].add_measurement(value)
            except Exception as e:
                logger.error(f"Error evaluating metric {metric_name}: {e}")
                measurements[metric_name] = 0.0
        
        # Record evaluation
        self.evaluation_history.append({
            "timestamp": datetime.now(),
            "goal": current_goal,
            "measurements": measurements.copy()
        })
        
        return measurements
    
    def get_overall_performance(self) -> float:
        """
        Calculate weighted overall performance score.
        
        This aggregates multiple metrics into a single performance
        indicator used for strategic goal evaluation.
        """
        if not self.metrics:
            return 0.5  # Default neutral performance
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric in self.metrics.values():
            if metric.history:
                weighted_sum += metric.get_recent_average() * metric.weight
                total_weight += metric.weight
        
        if total_weight == 0:
            return 0.5
        
        return weighted_sum / total_weight
    
    def get_strategy_performance(self) -> Dict[str, float]:
        """
        Get performance broken down by strategy.
        
        This provides the feedback signal for tactical strategy adaptation.
        """
        # For now, return overall performance for each metric
        # In a more sophisticated implementation, this would track
        # which strategies were active when each measurement was taken
        
        performance = {}
        for metric_name, metric in self.metrics.items():
            performance[metric_name] = metric.get_recent_average()
        
        return performance
    
    def get_value_assessments(self) -> Dict[str, Any]:
        """
        Get interpretable assessment of current performance.
        
        This provides detailed insights into what the agent is doing
        well and what needs improvement.
        """
        assessments = {}
        
        for metric_name, metric in self.metrics.items():
            assessment = {
                "current_value": metric.get_recent_average(),
                "trend": metric.get_trend(),
                "weight": metric.weight,
            }
            
            if metric.target_value is not None:
                assessment["target"] = metric.target_value
                assessment["gap"] = metric.target_value - metric.get_recent_average()
                assessment["progress"] = min(
                    1.0,
                    metric.get_recent_average() / metric.target_value
                ) if metric.target_value > 0 else 0.0
            
            assessments[metric_name] = assessment
        
        return {
            "overall_performance": self.get_overall_performance(),
            "metrics": assessments,
            "recent_evaluations": self.evaluation_history[-5:],
        }
    
    def reset_metrics(self):
        """Reset all metric histories."""
        for metric in self.metrics.values():
            metric.history.clear()
        self.evaluation_history.clear()
        logger.info("Value metrics reset")
