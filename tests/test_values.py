"""Tests for value module."""

import pytest
from gsv.values import ValueModule, ValueMetric


class TestValueMetric:
    """Test suite for ValueMetric class."""
    
    def test_metric_creation(self):
        """Test basic metric creation."""
        metric = ValueMetric(
            name="test_metric",
            description="A test metric",
            weight=1.0,
            target_value=0.8
        )
        
        assert metric.name == "test_metric"
        assert metric.description == "A test metric"
        assert metric.weight == 1.0
        assert metric.target_value == 0.8
    
    def test_metric_add_measurement(self):
        """Test adding measurements to metric."""
        metric = ValueMetric(name="test", description="Test")
        
        metric.add_measurement(0.5)
        metric.add_measurement(0.7)
        
        assert len(metric.history) == 2
        assert metric.history[-1] == 0.7
    
    def test_metric_history_limit(self):
        """Test that metric history is bounded."""
        metric = ValueMetric(name="test", description="Test")
        
        # Add many measurements
        for i in range(1100):
            metric.add_measurement(0.5)
        
        # History should be bounded to 1000
        assert len(metric.history) == 1000
    
    def test_metric_recent_average(self):
        """Test recent average calculation."""
        metric = ValueMetric(name="test", description="Test")
        
        metric.add_measurement(0.4)
        metric.add_measurement(0.6)
        metric.add_measurement(0.8)
        
        avg = metric.get_recent_average(window=3)
        assert abs(avg - 0.6) < 0.01
    
    def test_metric_trend_improving(self):
        """Test trend detection for improving metric."""
        metric = ValueMetric(name="test", description="Test")
        
        # Add older lower values
        for _ in range(10):
            metric.add_measurement(0.3)
        
        # Add recent higher values
        for _ in range(10):
            metric.add_measurement(0.7)
        
        trend = metric.get_trend(window=10)
        assert trend == "improving"
    
    def test_metric_trend_declining(self):
        """Test trend detection for declining metric."""
        metric = ValueMetric(name="test", description="Test")
        
        # Add older higher values
        for _ in range(10):
            metric.add_measurement(0.7)
        
        # Add recent lower values
        for _ in range(10):
            metric.add_measurement(0.3)
        
        trend = metric.get_trend(window=10)
        assert trend == "declining"
    
    def test_metric_trend_stable(self):
        """Test trend detection for stable metric."""
        metric = ValueMetric(name="test", description="Test")
        
        # Add stable values
        for _ in range(20):
            metric.add_measurement(0.5)
        
        trend = metric.get_trend(window=10)
        assert trend == "stable"
    
    def test_metric_trend_insufficient_data(self):
        """Test trend detection with insufficient data."""
        metric = ValueMetric(name="test", description="Test")
        
        metric.add_measurement(0.5)
        
        trend = metric.get_trend(window=10)
        assert trend == "insufficient_data"


class TestValueModule:
    """Test suite for ValueModule class."""
    
    def test_module_initialization(self):
        """Test value module can be initialized."""
        module = ValueModule()
        
        assert module is not None
        assert len(module.metrics) == 0
        assert len(module.evaluators) == 0
    
    def test_add_metric(self):
        """Test adding metrics to the module."""
        module = ValueModule()
        metric = ValueMetric(name="metric1", description="First")
        
        module.add_metric(metric)
        
        assert "metric1" in module.metrics
    
    def test_add_evaluator(self):
        """Test adding evaluator function."""
        module = ValueModule()
        
        def evaluator(obs, act, goal, ctx):
            return 0.5
        
        module.add_evaluator("metric1", evaluator)
        
        assert "metric1" in module.evaluators
    
    def test_evaluate(self):
        """Test evaluating performance."""
        module = ValueModule()
        
        metric = ValueMetric(name="metric1", description="First")
        module.add_metric(metric)
        
        def evaluator(obs, act, goal, ctx):
            return 0.7
        
        module.add_evaluator("metric1", evaluator)
        
        measurements = module.evaluate({}, {}, "goal1", {})
        
        assert "metric1" in measurements
        assert measurements["metric1"] == 0.7
        assert len(metric.history) == 1
    
    def test_evaluate_multiple_metrics(self):
        """Test evaluating multiple metrics."""
        module = ValueModule()
        
        metric1 = ValueMetric(name="metric1", description="First")
        metric2 = ValueMetric(name="metric2", description="Second")
        
        module.add_metric(metric1)
        module.add_metric(metric2)
        
        module.add_evaluator("metric1", lambda obs, act, goal, ctx: 0.6)
        module.add_evaluator("metric2", lambda obs, act, goal, ctx: 0.8)
        
        measurements = module.evaluate({}, {}, "goal1", {})
        
        assert len(measurements) == 2
        assert measurements["metric1"] == 0.6
        assert measurements["metric2"] == 0.8
    
    def test_evaluate_error_handling(self):
        """Test that evaluation errors are handled gracefully."""
        module = ValueModule()
        
        def failing_evaluator(obs, act, goal, ctx):
            raise Exception("Test error")
        
        module.add_evaluator("metric1", failing_evaluator)
        
        measurements = module.evaluate({}, {}, "goal1", {})
        
        # Should return 0.0 for failed metric
        assert measurements["metric1"] == 0.0
    
    def test_get_overall_performance_no_metrics(self):
        """Test overall performance with no metrics."""
        module = ValueModule()
        
        performance = module.get_overall_performance()
        
        assert performance == 0.5  # Default neutral
    
    def test_get_overall_performance(self):
        """Test overall performance calculation."""
        module = ValueModule()
        
        metric1 = ValueMetric(name="metric1", description="First", weight=1.0)
        metric2 = ValueMetric(name="metric2", description="Second", weight=2.0)
        
        module.add_metric(metric1)
        module.add_metric(metric2)
        
        # Add measurements
        metric1.add_measurement(0.5)
        metric2.add_measurement(0.8)
        
        performance = module.get_overall_performance()
        
        # Weighted average: (0.5 * 1.0 + 0.8 * 2.0) / 3.0 = 0.7
        assert abs(performance - 0.7) < 0.01
    
    def test_get_strategy_performance(self):
        """Test getting strategy-level performance."""
        module = ValueModule()
        
        metric = ValueMetric(name="metric1", description="First")
        module.add_metric(metric)
        
        metric.add_measurement(0.6)
        
        performance = module.get_strategy_performance()
        
        assert "metric1" in performance
        assert abs(performance["metric1"] - 0.6) < 0.01
    
    def test_get_value_assessments(self):
        """Test getting interpretable value assessments."""
        module = ValueModule()
        
        metric = ValueMetric(
            name="metric1",
            description="First",
            weight=1.0,
            target_value=0.8
        )
        module.add_metric(metric)
        
        # Add measurements
        for _ in range(10):
            metric.add_measurement(0.6)
        
        assessments = module.get_value_assessments()
        
        assert "overall_performance" in assessments
        assert "metrics" in assessments
        assert "metric1" in assessments["metrics"]
        
        metric_assessment = assessments["metrics"]["metric1"]
        assert "current_value" in metric_assessment
        assert "trend" in metric_assessment
        assert "target" in metric_assessment
        assert "gap" in metric_assessment
        assert "progress" in metric_assessment
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        module = ValueModule()
        
        metric = ValueMetric(name="metric1", description="First")
        module.add_metric(metric)
        
        metric.add_measurement(0.5)
        metric.add_measurement(0.7)
        
        module.reset_metrics()
        
        assert len(metric.history) == 0
        assert len(module.evaluation_history) == 0
