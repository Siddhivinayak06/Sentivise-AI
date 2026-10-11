"""Unit tests for Explainable AI (SHAP, LIME, Counterfactuals) and Report exports."""
import pytest
from services.explainability_service import ExplainabilityService
from services.report_service import ReportService

def test_shap_explanation_linear():
    review = "Great CD, wonderful performance and great sound."
    res = ExplainabilityService.explain_with_shap(review, model_name="Logistic Regression (Best)", top_k=5)
    assert res["success"]
    assert len(res["top_features"]) > 0
    assert "attribution" in res["top_features"][0]

def test_counterfactual_generation():
    review = "Terrible quality, broke after one day, waste of money."
    cf_res = ExplainabilityService.generate_counterfactual(review, model_name="Logistic Regression (Best)")
    assert cf_res["success"]
    assert cf_res["original_sentiment"] == "Negative"
    assert cf_res["target_sentiment"] == "Positive"
    assert len(cf_res["counterfactuals"]) > 0

def test_executive_reports_generation():
    json_rep = ReportService.generate_executive_report_json()
    assert "application" in json_rep
    assert "kpi_metrics" in json_rep

    md_rep = ReportService.generate_executive_report_markdown()
    assert "# Executive Intelligence" in md_rep
    assert "Total Customer Reviews Analyzed" in md_rep
