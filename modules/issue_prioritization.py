"""Issue Prioritization Module providing transparent, multi-factor priority scoring."""
from typing import Dict, Any, Tuple
from config import DEFAULT_PRIORITY_WEIGHTS

SEVERITY_SCORES = {
    "Critical": 100.0,
    "High": 75.0,
    "Medium": 50.0,
    "Low": 25.0
}

class IssuePrioritizer:
    """Calculates transparent multi-factor priority scores for customer issues."""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or dict(DEFAULT_PRIORITY_WEIGHTS)
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def calculate_priority(
        self,
        frequency: int,
        negative_ratio: float,
        severity: str,
        recency_days: int = 7,
        affected_products: int = 1,
        max_benchmark_frequency: int = 50,
        max_benchmark_products: int = 10
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Calculates a priority score on a 0-100 scale along with a factor breakdown and rationale.
        """
        # Factor 1: Frequency score (capped at 100)
        s_freq = min(100.0, (frequency / max(1, max_benchmark_frequency)) * 100.0)

        # Factor 2: Negative ratio score (0.0 to 1.0 -> 0 to 100)
        s_neg = min(100.0, max(0.0, negative_ratio * 100.0))

        # Factor 3: Severity score
        s_sev = SEVERITY_SCORES.get(severity, 50.0)

        # Factor 4: Recency score (shorter days = higher urgency)
        if recency_days <= 3:
            s_rec = 100.0
        elif recency_days <= 14:
            s_rec = 80.0
        elif recency_days <= 30:
            s_rec = 60.0
        elif recency_days <= 90:
            s_rec = 40.0
        else:
            s_rec = 20.0

        # Factor 5: Affected products score
        s_prod = min(100.0, (affected_products / max(1, max_benchmark_products)) * 100.0)

        # Weighted calculation
        score = (
            self.weights["frequency"] * s_freq +
            self.weights["negative_ratio"] * s_neg +
            self.weights["severity"] * s_sev +
            self.weights["recency"] * s_rec +
            self.weights["affected_products"] * s_prod
        )
        score = round(min(100.0, max(0.0, score)), 1)

        breakdown = {
            "frequency_factor": round(s_freq, 1),
            "negative_ratio_factor": round(s_neg, 1),
            "severity_factor": round(s_sev, 1),
            "recency_factor": round(s_rec, 1),
            "affected_products_factor": round(s_prod, 1),
            "weighted_score": score
        }

        # Human-readable rationale
        rationale_parts = []
        if s_sev >= 75:
            rationale_parts.append(f"high severity rating ({severity})")
        if frequency >= 10:
            rationale_parts.append(f"significant complaint volume ({frequency} reports)")
        if negative_ratio >= 0.70:
            rationale_parts.append(f"predominantly negative sentiment ({int(negative_ratio * 100)}%)")
        if affected_products > 1:
            rationale_parts.append(f"cross-product impact ({affected_products} distinct SKUs)")
        if recency_days <= 7:
            rationale_parts.append("active recency within the last week")

        if not rationale_parts:
            rationale_parts.append("baseline standard review telemetry")

        rationale = f"Priority {score}/100 driven by " + ", ".join(rationale_parts) + "."

        return score, breakdown, rationale
