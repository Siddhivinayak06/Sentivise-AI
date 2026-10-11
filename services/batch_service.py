"""Batch Processing Service for validating, analyzing, and storing bulk review datasets."""
from typing import Dict, Any, List, Tuple, Optional, Callable
import pandas as pd
import numpy as np

from modules.preprocessing import clean_text, build_feature_matrix
from services.model_loader import load_tfidf_vectorizer, load_selected_feature_names, load_ml_model
from services.prediction_service import extract_aspect_facts
from modules.knowledge_base import KnowledgeBase
from modules.rule_engine import ForwardChainingEngine
from database.repository import Repository
from config import LABEL_MAPPING

_kb = KnowledgeBase()
_forward_engine = ForwardChainingEngine(_kb)

def validate_batch_dataframe(
    df: pd.DataFrame,
    text_col: str,
    rating_col: Optional[str] = None,
    date_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validates uploaded DataFrame:
    - Verifies text column presence
    - Counts missing values
    - Detects exact duplicates
    - Validates rating range (e.g. 1.0 to 5.0)
    - Validates dates
    """
    total_rows = len(df)
    report = {
        "total_rows": total_rows,
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "missing_text_count": 0,
        "duplicate_count": 0,
        "invalid_rating_count": 0,
        "invalid_date_count": 0
    }

    if text_col not in df.columns:
        report["is_valid"] = False
        report["errors"].append(f"Required review text column '{text_col}' not found.")
        return report

    # Missing text
    missing_text = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    report["missing_text_count"] = int(missing_text.sum())

    # Duplicates in text
    report["duplicate_count"] = int(df.duplicated(subset=[text_col]).sum())

    # Rating validation
    if rating_col and rating_col in df.columns:
        numeric_ratings = pd.to_numeric(df[rating_col], errors='coerce')
        invalid_ratings = numeric_ratings.isna() & df[rating_col].notna()
        out_of_bounds = (numeric_ratings < 1.0) | (numeric_ratings > 5.0)
        report["invalid_rating_count"] = int((invalid_ratings | out_of_bounds).sum())

    # Date validation
    if date_col and date_col in df.columns:
        parsed_dates = pd.to_datetime(df[date_col], errors='coerce')
        report["invalid_date_count"] = int((parsed_dates.isna() & df[date_col].notna()).sum())

    return report

def process_batch_reviews(
    df: pd.DataFrame,
    text_col: str,
    product_id_col: Optional[str] = None,
    product_name_col: Optional[str] = None,
    date_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    model_name: str = "Logistic Regression (Best)",
    remove_duplicates: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    persist_to_db: bool = True,
    filename: str = "batch_upload.csv",
    chunk_size: int = 500
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes vectorized batch inference across uploaded reviews.
    Preserves original columns and appends analytical outputs.
    """
    initial_count = len(df)
    work_df = df.copy()

    if remove_duplicates:
        work_df = work_df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    dedup_count = len(work_df)

    tfidf = load_tfidf_vectorizer()
    selected_features = load_selected_feature_names()
    model = load_ml_model(model_name)

    if model is None or tfidf is None or not selected_features:
        raise ValueError("Model artifacts could not be loaded for batch inference.")

    # Prepare output containers
    predicted_sentiments: List[str] = []
    prediction_confidences: List[float] = []
    matched_rules_list: List[str] = []
    recommendations_list: List[str] = []
    analysis_statuses: List[str] = []

    total_chunks = max(1, (dedup_count + chunk_size - 1) // chunk_size)

    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(dedup_count, (chunk_idx + 1) * chunk_size)
        chunk_slice = work_df.iloc[start_idx:end_idx]

        raw_texts = chunk_slice[text_col].fillna("").astype(str).tolist()
        cleaned_texts = [clean_text(t) for t in raw_texts]

        # Valid rows filter
        valid_mask = [len(t) >= 3 for t in cleaned_texts]
        chunk_features = build_feature_matrix(cleaned_texts, tfidf, selected_features)

        # Batch prediction
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(chunk_features)
            preds = np.argmax(probs, axis=1)
            confs = [float(probs[i, preds[i]]) for i in range(len(preds))]
        else:
            preds = model.predict(chunk_features)
            confs = [1.0] * len(preds)

        for i, (is_valid, pred, conf, text) in enumerate(zip(valid_mask, preds, confs, cleaned_texts)):
            if not is_valid:
                predicted_sentiments.append("Invalid / Too Short")
                prediction_confidences.append(0.0)
                matched_rules_list.append("")
                recommendations_list.append("Review text empty or too short for classification.")
                analysis_statuses.append("Rejected")
                continue

            sent = LABEL_MAPPING.get(int(pred), "Unknown")
            predicted_sentiments.append(sent)
            prediction_confidences.append(round(conf, 4))
            analysis_statuses.append("Success")

            # Aspect facts & rule deduction
            facts = extract_aspect_facts(text)
            facts["sentiment"] = sent
            facts["confidence"] = conf
            rule_res = _forward_engine.infer(facts)

            rules_str = ", ".join(rule_res["fired_rules"]) if rule_res["fired_rules"] else "None"
            recs = [r["recommendation"] for r in rule_res["recommendations"]]
            rec_str = " | ".join(recs) if recs else ("Standard feedback - No escalation" if sent == "Positive" else "Monitor feedback trends")

            matched_rules_list.append(rules_str)
            recommendations_list.append(rec_str)

        if progress_callback:
            progress = (chunk_idx + 1) / total_chunks
            progress_callback(progress, f"Processed batch {chunk_idx + 1} of {total_chunks}...")

    # Attach columns to DataFrame
    work_df["predicted_sentiment"] = predicted_sentiments
    work_df["prediction_confidence"] = prediction_confidences
    work_df["selected_model"] = model_name
    work_df["matched_rules"] = matched_rules_list
    work_df["recommendation"] = recommendations_list
    work_df["analysis_status"] = analysis_statuses

    # Summary
    success_count = sum(1 for s in analysis_statuses if s == "Success")
    rejected_count = len(analysis_statuses) - success_count
    pos_count = sum(1 for s in predicted_sentiments if s == "Positive")
    neg_count = sum(1 for s in predicted_sentiments if s == "Negative")

    summary = {
        "initial_rows": initial_count,
        "analyzed_rows": dedup_count,
        "duplicates_removed": initial_count - dedup_count,
        "successful_predictions": success_count,
        "rejected_rows": rejected_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_percentage": round((pos_count / success_count * 100), 1) if success_count > 0 else 0.0,
        "negative_percentage": round((neg_count / success_count * 100), 1) if success_count > 0 else 0.0
    }

    # Persist in SQLite
    if persist_to_db:
        import_id = Repository.create_batch_import(
            filename=filename,
            total_rows=initial_count,
            valid_rows=success_count,
            invalid_rows=rejected_count + (initial_count - dedup_count)
        )
        db_records = []
        for idx, row in work_df.iterrows():
            db_records.append({
                "import_id": import_id,
                "original_index": int(idx),
                "review_text": str(row[text_col])[:1000],
                "product_id": str(row[product_id_col]) if product_id_col and product_id_col in row and pd.notna(row[product_id_col]) else None,
                "product_name": str(row[product_name_col]) if product_name_col and product_name_col in row and pd.notna(row[product_name_col]) else None,
                "review_date": str(row[date_col]) if date_col and date_col in row and pd.notna(row[date_col]) else None,
                "rating": float(row[rating_col]) if rating_col and rating_col in row and pd.notna(row[rating_col]) else None,
                "sentiment": row["predicted_sentiment"],
                "confidence": row["prediction_confidence"],
                "model_used": model_name,
                "rules_triggered": row["matched_rules"],
                "recommendation": row["recommendation"],
                "analysis_status": row["analysis_status"]
            })
        Repository.insert_batch_reviews(db_records)
        summary["import_id"] = import_id

    return work_df, summary
