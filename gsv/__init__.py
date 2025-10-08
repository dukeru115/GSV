"""
GSV: Goal-Strategy-Value Architecture for Autonomous AI Agents

A multi-timescale cognitive architecture that enables autonomous agents to operate
with both fast reactive cognition and strategic behavioral adaptation.
"""

from .core import GSVAgent, TimescaleLevel
from .goals import GoalModule, Goal
from .strategies import StrategyModule, Strategy
from .values import ValueModule, ValueMetric

__version__ = "0.1.0"
__all__ = [
    "GSVAgent",
    "TimescaleLevel",
    "GoalModule",
    "Goal",
    "StrategyModule",
    "Strategy",
    "ValueModule",
    "ValueMetric",
]
