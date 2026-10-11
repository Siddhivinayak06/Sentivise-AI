"""Unit tests for batch analytics validation, deduplication, and inference."""
import pytest
import pandas as pd
from services.batch_service import validate_batch_dataframe, process_batch_reviews

def test_validate_batch_dataframe_catches_missing():
    df = pd.DataFrame({
        "comment": ["Good product", "", None, "Another review"],
        "rating": [5, 4, 1, 9] # 9 is invalid rating
    })
    report = validate_batch_dataframe(df, text_col="comment", rating_col="rating")
    assert report["is_valid"]
    assert report["missing_text_count"] == 2
    assert report["invalid_rating_count"] == 1

def test_process_batch_reviews():
    df = pd.DataFrame({
        "review_text": [
            "This is wonderful, excellent performance and great quality!",
            "Terrible, broke immediately, completely waste of money.",
            "This is wonderful, excellent performance and great quality!" # duplicate
        ],
        "product_id": ["P1", "P2", "P1"],
        "rating": [5.0, 1.0, 5.0]
    })

    analyzed_df, summary = process_batch_reviews(
        df=df,
        text_col="review_text",
        product_id_col="product_id",
        rating_col="rating",
        model_name="Logistic Regression (Best)",
        remove_duplicates=True,
        persist_to_db=False
    )

    assert len(analyzed_df) == 2  # 1 duplicate removed
    assert summary["duplicates_removed"] == 1
    assert "predicted_sentiment" in analyzed_df.columns
    assert "prediction_confidence" in analyzed_df.columns
    assert "matched_rules" in analyzed_df.columns
    assert "recommendation" in analyzed_df.columns

    # First row is positive, second is negative
    assert analyzed_df.iloc[0]["predicted_sentiment"] == "Positive"
    assert analyzed_df.iloc[1]["predicted_sentiment"] == "Negative"
