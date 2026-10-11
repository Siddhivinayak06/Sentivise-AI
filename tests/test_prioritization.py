"""Unit tests for transparent issue prioritization scoring and rationale calculations."""
import pytest
from modules.issue_prioritization import IssuePrioritizer

def test_prioritizer_weight_normalization():
    # Weights that don't sum to 1.0
    unnormalized = {"frequency": 1.0, "negative_ratio": 1.0, "severity": 1.0, "recency": 1.0, "affected_products": 1.0}
    p = IssuePrioritizer(unnormalized)
    assert sum(p.weights.values()) == pytest.approx(1.0, 1e-4)
    assert p.weights["frequency"] == pytest.approx(0.2, 1e-4)

def test_prioritizer_critical_scoring():
    p = IssuePrioritizer()
    score, breakdown, rationale = p.calculate_priority(
        frequency=50,
        negative_ratio=1.0,
        severity="Critical",
        recency_days=1,
        affected_products=10
    )
    # At maximal inputs, score should be near 100
    assert score >= 90.0
    assert "Critical" in breakdown["severity_factor"].__str__() or breakdown["severity_factor"] == 100.0
    assert "high severity" in rationale.lower()

def test_prioritizer_low_urgency_scoring():
    p = IssuePrioritizer()
    score, breakdown, rationale = p.calculate_priority(
        frequency=1,
        negative_ratio=0.1,
        severity="Low",
        recency_days=120,
        affected_products=1
    )
    assert score <= 35.0
