"""Text preprocessing and feature engineering module.
Replicates and unifies the exact feature generation logic from Task 5, Task 6, and Task 8.
"""
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from config import POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

POS_KEYWORD_SET = set(POSITIVE_KEYWORDS)
NEG_KEYWORD_SET = set(NEGATIVE_KEYWORDS)

def clean_text(text: str) -> str:
    """Standardizes review text: removes abnormal whitespaces and handles empty inputs."""
    if not isinstance(text, str):
        return ""
    # Strip whitespace, collapse internal multiple spaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def validate_review_input(text: str, min_chars: int = 3, max_chars: int = 5000) -> Tuple[bool, Optional[str]]:
    """Validates user review text before model inference."""
    if text is None:
        return False, "Review text cannot be null."
    cleaned = clean_text(text)
    if len(cleaned) < min_chars:
        return False, f"Review text is too short (minimum {min_chars} characters required)."
    if len(cleaned) > max_chars:
        return False, f"Review text exceeds maximum allowed limit of {max_chars} characters."
    return True, None

def extract_lexical_and_structural_features(texts: List[str]) -> pd.DataFrame:
    """
    Extracts the 12 engineered features for a list of review texts:
    1. review_length
    2. word_count
    3. character_count
    4. sentence_count
    5. average_word_length
    6. exclamation_count
    7. question_count
    8. uppercase_count
    9. digit_count
    10. positive_word_count
    11. negative_word_count
    12. sentiment_keyword_score
    """
    records = []
    for t in texts:
        s_txt = str(t) if t is not None else ""
        length = len(s_txt)
        words = s_txt.split()
        word_count = len(words)
        char_count = length
        sentences = s_txt.count('.') + s_txt.count('!') + s_txt.count('?') + 1
        avg_word_len = (char_count / word_count) if word_count > 0 else 0.0
        exclamation_count = s_txt.count('!')
        question_count = s_txt.count('?')
        uppercase_count = sum(1 for c in s_txt if c.isupper())
        digit_count = sum(1 for c in s_txt if c.isdigit())

        lower_words = s_txt.lower().split()
        pos_count = sum(1 for w in lower_words if w in POS_KEYWORD_SET)
        neg_count = sum(1 for w in lower_words if w in NEG_KEYWORD_SET)
        keyword_score = pos_count - neg_count

        records.append([
            length, word_count, char_count, sentences,
            avg_word_len, exclamation_count, question_count,
            uppercase_count, digit_count, pos_count,
            neg_count, keyword_score
        ])

    columns = [
        'review_length', 'word_count', 'character_count', 'sentence_count',
        'average_word_length', 'exclamation_count', 'question_count',
        'uppercase_count', 'digit_count', 'positive_word_count',
        'negative_word_count', 'sentiment_keyword_score'
    ]
    return pd.DataFrame(records, columns=columns)

def build_feature_matrix(
    texts: List[str],
    tfidf_vectorizer: Any,
    selected_feature_names: List[str]
) -> csr_matrix:
    """
    Builds the exact 100-dimensional sparse feature matrix expected by
    Task 6 ML models and Task 8 explainability tools.
    """
    # 1. Structural and Lexical features
    eng_df = extract_lexical_and_structural_features(texts)

    # 2. TF-IDF features
    tfidf_sparse = tfidf_vectorizer.transform([str(t) if t is not None else "" for t in texts])

    # 3. Combine column names to identify selected indices
    all_feature_names = list(eng_df.columns) + list(tfidf_vectorizer.get_feature_names_out())
    indices = [all_feature_names.index(f) for f in selected_feature_names if f in all_feature_names]

    # 4. Stack horizontally
    full_sparse = hstack([csr_matrix(eng_df.values), tfidf_sparse])

    # 5. Slice down to the exact selected 100 features
    return full_sparse[:, indices].tocsr()

def build_dnn_feature_matrix(
    texts: List[str],
    tfidf_vectorizer: Any,
    selected_feature_names: List[str],
    dnn_scaler: Any
) -> np.ndarray:
    """
    Builds dense, float32, standardized feature matrix expected by Task 7 DNN model.
    """
    sparse_matrix = build_feature_matrix(texts, tfidf_vectorizer, selected_feature_names)
    dense_features = sparse_matrix.toarray().astype(np.float32)
    dense_features = np.nan_to_num(dense_features)
    scaled_features = dnn_scaler.transform(dense_features).astype(np.float32)
    return scaled_features
