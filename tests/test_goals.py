"""Tests for goal module."""

import pytest
from gsv.goals import GoalModule, Goal


class TestGoal:
    """Test suite for Goal class."""
    
    def test_goal_creation(self):
        """Test basic goal creation."""
        goal = Goal(
            name="test_goal",
            description="A test goal",
            priority=1.0,
            success_threshold=0.8
        )
        
        assert goal.name == "test_goal"
        assert goal.description == "A test goal"
        assert goal.priority == 1.0
        assert goal.success_threshold == 0.8
    
    def test_goal_success_evaluation(self):
        """Test goal success evaluation."""
        goal = Goal(
            name="test_goal",
            description="Test",
            success_threshold=0.8
        )
        
        assert goal.evaluate_success(0.9) is True
        assert goal.evaluate_success(0.8) is True
        assert goal.evaluate_success(0.7) is False


class TestGoalModule:
    """Test suite for GoalModule class."""
    
    def test_module_initialization(self):
        """Test goal module can be initialized."""
        module = GoalModule()
        assert module is not None
        assert len(module.goals) == 0
        assert module.active_goal is None
    
    def test_add_goal(self):
        """Test adding goals to the module."""
        module = GoalModule()
        goal = Goal(name="goal1", description="First goal")
        
        module.add_goal(goal)
        
        assert "goal1" in module.goals
        assert module.active_goal == "goal1"  # First goal becomes active
    
    def test_add_multiple_goals(self):
        """Test adding multiple goals."""
        module = GoalModule()
        
        goal1 = Goal(name="goal1", description="First")
        goal2 = Goal(name="goal2", description="Second")
        
        module.add_goal(goal1)
        module.add_goal(goal2)
        
        assert len(module.goals) == 2
        assert module.active_goal == "goal1"  # First added stays active
    
    def test_set_active_goal(self):
        """Test manually setting active goal."""
        module = GoalModule()
        
        goal1 = Goal(name="goal1", description="First")
        goal2 = Goal(name="goal2", description="Second")
        
        module.add_goal(goal1)
        module.add_goal(goal2)
        
        module.set_active_goal("goal2")
        
        assert module.active_goal == "goal2"
        assert len(module.goal_history) > 0
    
    def test_set_active_goal_invalid(self):
        """Test setting non-existent goal raises error."""
        module = GoalModule()
        
        with pytest.raises(ValueError):
            module.set_active_goal("nonexistent")
    
    def test_evaluate_goals_no_goal(self):
        """Test evaluating when no goals are set."""
        module = GoalModule()
        
        updates = module.evaluate_goals(0.5, {}, {})
        
        assert len(updates) == 0
    
    def test_evaluate_goals_achievement(self):
        """Test goal achievement detection."""
        module = GoalModule()
        
        goal1 = Goal(name="goal1", description="First", success_threshold=0.8)
        goal2 = Goal(name="goal2", description="Second", priority=0.9)
        
        module.add_goal(goal1)
        module.add_goal(goal2)
        
        # Achieve goal1
        updates = module.evaluate_goals(0.9, {}, {})
        
        assert "goal_achieved" in updates
        assert updates["goal_achieved"] == "goal1"
        assert "new_goal" in updates
        assert module.active_goal == "goal2"
    
    def test_get_active_goal(self):
        """Test retrieving active goal."""
        module = GoalModule()
        
        assert module.get_active_goal() is None
        
        goal = Goal(name="goal1", description="First")
        module.add_goal(goal)
        
        assert module.get_active_goal() == "goal1"
    
    def test_get_goal_details(self):
        """Test retrieving goal details."""
        module = GoalModule()
        
        goal = Goal(
            name="goal1",
            description="Test goal",
            priority=1.0,
            success_threshold=0.8
        )
        module.add_goal(goal)
        
        details = module.get_goal_details()
        
        assert "active_goal" in details
        assert "goals" in details
        assert "goal1" in details["goals"]
        assert details["goals"]["goal1"]["description"] == "Test goal"
    
    def test_remove_goal(self):
        """Test removing a goal."""
        module = GoalModule()
        
        goal1 = Goal(name="goal1", description="First")
        goal2 = Goal(name="goal2", description="Second")
        
        module.add_goal(goal1)
        module.add_goal(goal2)
        
        module.remove_goal("goal1")
        
        assert "goal1" not in module.goals
        assert module.active_goal == "goal2"  # Switched to next goal
    
    def test_remove_nonactive_goal(self):
        """Test removing a non-active goal."""
        module = GoalModule()
        
        goal1 = Goal(name="goal1", description="First")
        goal2 = Goal(name="goal2", description="Second")
        
        module.add_goal(goal1)
        module.add_goal(goal2)
        
        module.remove_goal("goal2")
        
        assert "goal2" not in module.goals
        assert module.active_goal == "goal1"  # Active goal unchanged
