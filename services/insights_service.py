"""Customer Insights Service providing theme-based complaint detection and term frequency analysis."""
import re
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

from config import ASPECT_TAXONOMY, POSITIVE_KEYWORDS, NEGATIVE_KEYWORDS

STOPWORDS = {
    "the", "and", "a", "to", "of", "is", "in", "it", "i", "that", "this", "for", "my",
    "with", "on", "was", "but", "as", "have", "you", "not", "so", "be", "are", "at",
    "one", "all", "very", "from", "by", "or", "an", "had", "they", "if", "more", "out",
    "would", "about", "up", "there", "what", "which", "when", "can", "their", "just",
    "them", "then", "into", "only", "no", "other", "some", "time", "could", "these"
}

def analyze_term_frequencies(reviews: List[str], top_n: int = 15) -> Dict[str, List[Tuple[str, int]]]:
    """Extracts top positive, negative, and general terms from review texts."""
    pos_counter = Counter()
    neg_counter = Counter()
    general_counter = Counter()

    pos_set = set(POSITIVE_KEYWORDS)
    neg_set = set(NEGATIVE_KEYWORDS)

    for text in reviews:
        if not isinstance(text, str):
            continue
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        for w in words:
            if w in STOPWORDS:
                continue
            general_counter[w] += 1
            if w in pos_set:
                pos_counter[w] += 1
            elif w in neg_set:
                neg_counter[w] += 1

    return {
        "top_positive_terms": pos_counter.most_common(top_n),
        "top_negative_terms": neg_counter.most_common(top_n),
        "top_general_terms": general_counter.most_common(top_n)
    }

def categorize_complaint_themes(
    df: pd.DataFrame,
    text_col: str = "review_text",
    sentiment_col: str = "predicted_sentiment"
) -> Dict[str, Any]:
    """
    Categorizes reviews into 5 core customer feedback themes using heuristic keywords:
    - Product Quality
    - Delivery & Logistics
    - Price & Value
    - Customer Support
    - Usability & Features
    """
    if df.empty or text_col not in df.columns:
        return {
            "theme_counts": {},
            "theme_reviews": {},
            "theme_sentiment_breakdown": {}
        }

    theme_counts = {k: 0 for k in ASPECT_TAXONOMY.keys()}
    theme_negative_counts = {k: 0 for k in ASPECT_TAXONOMY.keys()}
    theme_reviews = {k: [] for k in ASPECT_TAXONOMY.keys()}

    has_sentiment = sentiment_col in df.columns

    for _, row in df.iterrows():
        text = str(row[text_col])
        lower = text.lower()
        words = set(re.findall(r'\b[a-z]+\b', lower))
        is_neg = (str(row[sentiment_col]).lower() == "negative") if has_sentiment else False

        for theme, keywords in ASPECT_TAXONOMY.items():
            if any(k in words for k in keywords):
                theme_counts[theme] += 1
                if is_neg:
                    theme_negative_counts[theme] += 1
                if len(theme_reviews[theme]) < 5:
                    theme_reviews[theme].append({
                        "text": text[:250] + "..." if len(text) > 250 else text,
                        "sentiment": str(row[sentiment_col]) if has_sentiment else "N/A",
                        "product": str(row.get("product_id", row.get("product_name", "N/A")))
                    })

    # Sentiment breakdown
    theme_breakdown = []
    for theme in ASPECT_TAXONOMY.keys():
        total = theme_counts[theme]
        neg = theme_negative_counts[theme]
        pos = total - neg
        theme_breakdown.append({
            "Theme": theme,
            "Total Mentions": total,
            "Negative Count": neg,
            "Positive/Neutral Count": pos,
            "Negative Percentage": round((neg / total * 100), 1) if total > 0 else 0.0
        })

    return {
        "theme_counts": theme_counts,
        "theme_reviews": theme_reviews,
        "theme_breakdown_df": pd.DataFrame(theme_breakdown)
    }

def detect_repeated_issues(reviews: List[str], min_count: int = 3) -> List[Dict[str, Any]]:
    """Detects repeated 2-word complaint n-gram patterns in negative reviews."""
    phrase_counter = Counter()
    for text in reviews:
        if not isinstance(text, str):
            continue
        words = [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in STOPWORDS]
        for i in range(len(words) - 1):
            pair = f"{words[i]} {words[i+1]}"
            phrase_counter[pair] += 1

    repeated = []
    for phrase, count in phrase_counter.most_common(20):
        if count >= min_count:
            repeated.append({
                "phrase": phrase,
                "frequency": count,
                "theme_hint": _get_theme_hint(phrase)
            })
    return repeated

def _get_theme_hint(phrase: str) -> str:
    for theme, keywords in ASPECT_TAXONOMY.items():
        if any(k in phrase for k in keywords):
            return theme
    return "General Feedback"
