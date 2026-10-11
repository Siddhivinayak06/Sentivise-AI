"""Configuration module for AI Review Sentiment Analysis and Decision Support System."""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_ML_DIR = BASE_DIR / "models" / "ml"
MODELS_DNN_DIR = BASE_DIR / "models" / "dnn"
RESULTS_DIR = BASE_DIR / "results"
REPORTS_DIR = BASE_DIR / "results" / "reports"
FIGURES_DIR = BASE_DIR / "results" / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
SCRIPTS_DIR = BASE_DIR / "scripts"
DATABASE_PATH = BASE_DIR / "sentiment_system.db"
TASK8_DIR = BASE_DIR / "task8"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, MODELS_ML_DIR, MODELS_DNN_DIR, RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR]:
    os.makedirs(d, exist_ok=True)

def find_file(*candidates) -> Path:
    """Finds the first existing path among candidates, or returns the first."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])

# Dataset file paths
CLEAN_REVIEWS_CSV = find_file(DATA_DIR / "clean_product_reviews.csv", BASE_DIR / "clean_product_reviews.csv")
X_TRAIN_CSV = find_file(DATA_DIR / "X_train.csv", BASE_DIR / "X_train.csv")
X_TEST_CSV = find_file(DATA_DIR / "X_test.csv", BASE_DIR / "X_test.csv")

# Preprocessing & Model Artifact Paths
TFIDF_VECTORIZER_PATH = find_file(MODELS_ML_DIR / "task5_tfidf_vectorizer.pkl", BASE_DIR / "task5_tfidf_vectorizer.pkl")
SELECTED_FEATURES_PATH = find_file(MODELS_ML_DIR / "task5_selected_features.csv", BASE_DIR / "task5_selected_features.csv")
BEST_MODEL_PATH = find_file(MODELS_ML_DIR / "best_sentiment_model.pkl", BASE_DIR / "best_sentiment_model.pkl")

ML_MODEL_PATHS = {
    "Logistic Regression (Best)": find_file(MODELS_ML_DIR / "best_sentiment_model.pkl", BASE_DIR / "best_sentiment_model.pkl"),
    "Decision Tree": find_file(MODELS_ML_DIR / "decision_tree_model.pkl", BASE_DIR / "decision_tree_model.pkl"),
    "Random Forest": find_file(MODELS_ML_DIR / "random_forest_model.pkl", BASE_DIR / "random_forest_model.pkl"),
    "Support Vector Machine (SVM)": find_file(MODELS_ML_DIR / "svm_model.pkl", BASE_DIR / "svm_model.pkl"),
    "XGBoost": find_file(MODELS_ML_DIR / "xgboost_model.pkl", BASE_DIR / "xgboost_model.pkl"),
}

DNN_MODEL_PATH = find_file(MODELS_DNN_DIR / "task7_dnn_model.keras", BASE_DIR / "task7_dnn_model.keras")
DNN_MODEL_H5_PATH = find_file(MODELS_DNN_DIR / "task7_dnn_model.h5", BASE_DIR / "task7_dnn_model.h5")
DNN_SCALER_PATH = find_file(MODELS_DNN_DIR / "task7_scaler.pkl", BASE_DIR / "task7_scaler.pkl")
DNN_LABEL_ENCODER_PATH = find_file(MODELS_DNN_DIR / "task7_label_encoder.pkl", BASE_DIR / "task7_label_encoder.pkl")

# Evaluation Reports
TASK6_EVAL_REPORT_PATH = find_file(REPORTS_DIR / "task6_evaluation_report.csv", BASE_DIR / "task6_evaluation_report.csv")
TASK6_COMPARISON_REPORT_PATH = find_file(REPORTS_DIR / "task6_model_comparison_report.csv", BASE_DIR / "task6_model_comparison_report.csv")
TASK7_EVAL_REPORT_PATH = find_file(REPORTS_DIR / "task7_evaluation_report.csv", BASE_DIR / "task7_evaluation_report.csv")
TASK7_HISTORY_PATH = find_file(REPORTS_DIR / "task7_training_history.csv", BASE_DIR / "task7_training_history.csv")
TASK6_VS_TASK7_PATH = find_file(REPORTS_DIR / "task6_vs_task7_comparison.csv", BASE_DIR / "task6_vs_task7_comparison.csv")

# Label Mapping verified from clean_product_reviews.csv
# 0 -> Negative (__label__1)
# 1 -> Positive (__label__2)
LABEL_MAPPING = {
    0: "Negative",
    1: "Positive"
}
LABEL_RAW_MAPPING = {
    "__label__1": 0,
    "__label__2": 1
}

# Lexical Keywords (from Task 5 feature engineering)
POSITIVE_KEYWORDS = [
    'good', 'great', 'best', 'excellent', 'love', 'awesome', 'nice', 'perfect', 'amazing', 'happy'
]
NEGATIVE_KEYWORDS = [
    'bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'waste', 'disappointing', 'boring', 'slow'
]

# Aspect Definitions for Customer Insights & Rule Engine
ASPECT_TAXONOMY = {
    "Product Quality": ["quality", "defect", "broke", "broken", "durability", "material", "faulty", "cheap", "sturdy", "reliable"],
    "Delivery & Logistics": ["delivery", "shipping", "shipped", "arrived", "late", "delay", "carrier", "transit"],
    "Packaging": ["package", "packaging", "box", "bubble", "wrap", "crushed", "torn", "smashed", "container"],
    "Price & Value": ["price", "cost", "expensive", "money", "worth", "waste", "value", "refund", "overpriced", "cheap"],
    "Customer Support": ["service", "support", "rep", "representative", "contact", "helpline", "email", "response", "unhelpful"],
    "Usability & Features": ["usability", "instruction", "manual", "setup", "feature", "battery", "size", "fit", "button", "screen"]
}

# Decision Center Default Weights
DEFAULT_PRIORITY_WEIGHTS = {
    "frequency": 0.25,
    "negative_ratio": 0.25,
    "severity": 0.20,
    "recency": 0.15,
    "affected_products": 0.15
}

# UI Theme Colors (High-Contrast Modern Dark Enterprise Theme)
THEME_COLORS = {
    "primary": "#6366F1",          # Electric Indigo 500
    "primary_light": "#818CF8",    # Bright Indigo 400
    "secondary": "#06B6D4",        # Vivid Cyan 500
    "secondary_light": "#38BDF8",  # Sky 400
    "accent": "#10B981",           # Emerald 500
    "background": "#0B0F19",       # Deep Midnight Slate Canvas
    "surface": "#131B2E",          # Translucent Dark Glass Surface
    "card_bg": "#131B2E",          # Slate 850
    "card_bg_elevated": "#1E293B", # Slate 800
    "text": "#F8FAFC",             # Crisp Pure Off-White
    "text_secondary": "#CBD5E1",   # Light Slate
    "text_muted": "#94A3B8",       # Slate 400
    "border": "rgba(255, 255, 255, 0.08)",       # Modern subtle hairline border
    "border_subtle": "rgba(255, 255, 255, 0.05)",
    "border_active": "rgba(99, 102, 241, 0.45)", # Glowing Indigo Border
    "positive": "#10B981",         # Vivid Emerald 500
    "negative": "#F43F5E",         # Vivid Rose 500
    "warning": "#F59E0B",          # Vivid Amber 500
    "info": "#3B82F6",             # Electric Blue 500
    "gradient_start": "#6366F1",   # Indigo 500
    "gradient_end": "#8B5CF6",     # Violet 500
    "glow": "rgba(99, 102, 241, 0.2)",  # Indigo glow
}

# App Metadata
APP_TITLE = "Sentivise AI"
APP_SUBTITLE = "Enterprise Customer Feedback Intelligence & Automated Decision Platform"
APP_VERSION = "2.0.0"
