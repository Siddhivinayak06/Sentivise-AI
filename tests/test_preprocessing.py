"""Unit tests for text preprocessing, cleaning, and feature engineering."""
import pytest
import pandas as pd
import numpy as np
from modules.preprocessing import (
    clean_text, validate_review_input,
    extract_lexical_and_structural_features,
    build_feature_matrix
)
from services.model_loader import load_tfidf_vectorizer, load_selected_feature_names

def test_clean_text_normalizes_spaces():
    raw = "  This   is   a   test \n review with \t whitespace.  "
    cleaned = clean_text(raw)
    assert cleaned == "This is a test review with whitespace."

def test_clean_text_handles_non_strings():
    assert clean_text(None) == ""
    assert clean_text(123) == ""

def test_validate_review_input_rejects_empty():
    valid, msg = validate_review_input("")
    assert not valid
    assert "too short" in msg.lower()

    valid, msg = validate_review_input("   ")
    assert not valid
    assert "too short" in msg.lower()

def test_validate_review_input_accepts_valid():
    valid, msg = validate_review_input("Great product, loved using it!")
    assert valid
    assert msg is None

def test_extract_lexical_and_structural_features():
    texts = [
        "Great CD! Love it.",
        "Worst purchase ever. Broke in 2 days!"
    ]
    eng_df = extract_lexical_and_structural_features(texts)
    assert len(eng_df) == 2
    assert "review_length" in eng_df.columns
    assert "sentiment_keyword_score" in eng_df.columns
    assert "positive_word_count" in eng_df.columns
    assert "negative_word_count" in eng_df.columns

    # First text has 'great' and 'love' -> positive_word_count >= 2
    assert eng_df.iloc[0]["positive_word_count"] >= 2
    # Second text has 'worst' -> negative_word_count >= 1
    assert eng_df.iloc[1]["negative_word_count"] >= 1

def test_build_feature_matrix_shape():
    tfidf = load_tfidf_vectorizer()
    selected_features = load_selected_feature_names()
    assert tfidf is not None, "TF-IDF vectorizer artifact missing"
    assert len(selected_features) == 100, "Expected 100 selected features"

    sample_texts = ["Excellent build and great battery.", "Poor quality, stopped working."]
    matrix = build_feature_matrix(sample_texts, tfidf, selected_features)
    assert matrix.shape == (2, 100)
