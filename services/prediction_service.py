"""Prediction Service coordinating preprocessing, model inference, and rule chaining."""
import re
from typing import Dict, Any, List, Optional
import numpy as np

from config import LABEL_MAPPING, ASPECT_TAXONOMY
from modules.preprocessing import clean_text, validate_review_input, build_feature_matrix, build_dnn_feature_matrix
from modules.knowledge_base import KnowledgeBase
from modules.rule_engine import ForwardChainingEngine
from services.model_loader import (
    load_tfidf_vectorizer, load_selected_feature_names,
    load_ml_model, load_dnn_artifacts
)

# Global shared Knowledge Base and Rule Engine
_kb = KnowledgeBase()
_forward_engine = ForwardChainingEngine(_kb)

SAFETY_KEYWORDS = {"fire", "burn", "burning", "smoke", "explode", "exploded", "explosion", "shock", "electric", "choke", "hazard", "poison"}
COUNTERFEIT_KEYWORDS = {"fake", "counterfeit", "replica", "knockoff", "bootleg", "bogus"}

def extract_aspect_facts(text: str) -> Dict[str, Any]:
    """Extracts heuristic domain aspect facts and risk flags from text."""
    lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', lower))

    facts = {
        "safety_hazard_detected": bool(words & SAFETY_KEYWORDS),
        "counterfeit_risk": bool(words & COUNTERFEIT_KEYWORDS),
        "aspect_quality": bool(words & set(ASPECT_TAXONOMY["Product Quality"])),
        "aspect_delivery": bool(words & set(ASPECT_TAXONOMY["Delivery & Logistics"])),
        "aspect_packaging": bool(words & set(ASPECT_TAXONOMY["Packaging"])),
        "aspect_price": bool(words & set(ASPECT_TAXONOMY["Price & Value"])),
        "aspect_support": bool(words & set(ASPECT_TAXONOMY["Customer Support"])),
        "aspect_usability": bool(words & set(ASPECT_TAXONOMY["Usability & Features"])),
    }
    return facts

def predict_single_review(
    review_text: str,
    model_name: str = "Logistic Regression (Best)",
    run_rules: bool = True
) -> Dict[str, Any]:
    """
    Executes end-to-end inference for a single review:
    1. Validation
    2. Vectorization & Feature Selection
    3. Model Inference (ML or DNN)
    4. Calibrated Probabilities & Uncertainty Check
    5. Rule Engine Deduction
    6. Feature Attribution (Top Influencing Tokens)
    """
    valid, err_msg = validate_review_input(review_text)
    if not valid:
        return {
            "success": False,
            "error": err_msg,
            "review_text": review_text,
            "sentiment": "Invalid Input",
            "confidence": 0.0,
            "prob_pos": 0.0,
            "prob_neg": 0.0,
            "model_used": model_name,
            "uncertain": True,
            "rules_triggered": [],
            "recommendations": []
        }

    cleaned = clean_text(review_text)
    tfidf = load_tfidf_vectorizer()
    selected_features = load_selected_feature_names()

    if tfidf is None or not selected_features:
        return {
            "success": False,
            "error": "TF-IDF vectorizer or selected feature list could not be loaded.",
            "review_text": review_text
        }

    # Perform prediction
    try:
        if model_name == "Deep Neural Network (DNN)":
            dnn_art = load_dnn_artifacts()
            if not dnn_art["is_ready"]:
                return {
                    "success": False,
                    "error": "Deep Neural Network model or scaler artifact is missing.",
                    "review_text": review_text
                }
            scaled_feat = build_dnn_feature_matrix([cleaned], tfidf, selected_features, dnn_art["scaler"])
            prob_raw = float(dnn_art["model"].predict(scaled_feat, verbose=0)[0][0])
            prob_pos = prob_raw
            prob_neg = 1.0 - prob_pos
            pred_class = 1 if prob_pos >= 0.5 else 0
        else:
            model = load_ml_model(model_name)
            if model is None:
                return {
                    "success": False,
                    "error": f"Model artifact for '{model_name}' could not be loaded.",
                    "review_text": review_text
                }
            feat_matrix = build_feature_matrix([cleaned], tfidf, selected_features)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat_matrix)[0]
                prob_neg = float(probs[0])
                prob_pos = float(probs[1])
                pred_class = int(np.argmax(probs))
            else:
                pred_class = int(model.predict(feat_matrix)[0])
                prob_pos = 1.0 if pred_class == 1 else 0.0
                prob_neg = 1.0 - prob_pos

        sentiment = LABEL_MAPPING.get(pred_class, "Unknown")
        confidence = prob_pos if pred_class == 1 else prob_neg

        # Uncertainty check: if probability is between 0.45 and 0.55
        is_uncertain = (0.45 <= prob_pos <= 0.55)

        # Extract domain aspect facts
        facts = extract_aspect_facts(cleaned)
        facts["sentiment"] = sentiment
        facts["confidence"] = confidence
        facts["prob_pos"] = prob_pos
        facts["prob_neg"] = prob_neg
        facts["high_severity"] = bool(facts["safety_hazard_detected"] or (confidence >= 0.85 and sentiment == "Negative"))

        # Run forward chaining rule engine
        rule_results = {"fired_rules": [], "recommendations": [], "trace": []}
        if run_rules:
            rule_results = _forward_engine.infer(facts)

        # Extract local contributing tokens (from review text matching selected unigrams)
        influential_words = []
        tokens = cleaned.lower().split()
        for tok in tokens:
            if tok in selected_features:
                influential_words.append(tok)

        return {
            "success": True,
            "review_text": review_text,
            "cleaned_text": cleaned,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "prob_pos": round(prob_pos, 4),
            "prob_neg": round(prob_neg, 4),
            "model_used": model_name,
            "uncertain": is_uncertain,
            "status": "Review Manually" if is_uncertain else "Classified",
            "facts": facts,
            "rules_triggered": rule_results["fired_rules"],
            "recommendations": rule_results["recommendations"],
            "rule_trace": rule_results["trace"],
            "influential_words": list(dict.fromkeys(influential_words))[:10]
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Inference pipeline error: {str(e)}",
            "review_text": review_text,
            "sentiment": "Error",
            "confidence": 0.0,
            "model_used": model_name
        }
