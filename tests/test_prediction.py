"""Unit tests for ML and DNN model loading, prediction formatting, and uncertainty handling."""
import pytest
from services.model_loader import load_ml_model, get_model_readiness_status
from services.prediction_service import predict_single_review
from config import ML_MODEL_PATHS

def test_model_readiness_status():
    status = get_model_readiness_status()
    assert len(status) >= 5
    assert "Logistic Regression (Best)" in status
    assert status["Logistic Regression (Best)"]["ready"]

def test_load_all_ml_models():
    for name in ML_MODEL_PATHS.keys():
        model = load_ml_model(name)
        assert model is not None, f"Failed to load {name}"

def test_predict_single_review_positive():
    text = "This product is absolutely amazing! Great quality, fast performance, and I love it."
    result = predict_single_review(text, model_name="Logistic Regression (Best)")
    assert result["success"]
    assert result["sentiment"] == "Positive"
    assert result["confidence"] >= 0.70
    assert result["prob_pos"] >= 0.70
    assert result["prob_neg"] <= 0.30
    assert isinstance(result["rules_triggered"], list)

def test_predict_single_review_negative():
    text = "Terrible quality. Completely useless, broke on day one, and a total waste of money."
    result = predict_single_review(text, model_name="Logistic Regression (Best)")
    assert result["success"]
    assert result["sentiment"] == "Negative"
    assert result["confidence"] >= 0.70
    assert result["prob_neg"] >= 0.70
    assert result["prob_pos"] <= 0.30

def test_predict_empty_or_invalid_review():
    result = predict_single_review("   ", model_name="Logistic Regression (Best)")
    assert not result["success"]
    assert "error" in result

    result_short = predict_single_review("hi", model_name="Logistic Regression (Best)")
    assert not result_short["success"]
    assert "too short" in result_short["error"].lower()

def test_predict_multi_model_consistency():
    text = "Very good device, works nicely and durable."
    for m_name in ["Logistic Regression (Best)", "Random Forest", "Support Vector Machine (SVM)"]:
        res = predict_single_review(text, model_name=m_name)
        assert res["success"], f"Failed for {m_name}"
        assert res["sentiment"] in ["Positive", "Negative"]
