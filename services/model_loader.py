"""Model Loader Service caching model artifacts safely using Streamlit."""
import os
import joblib
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import streamlit as st
from config import (
    TFIDF_VECTORIZER_PATH, SELECTED_FEATURES_PATH, BEST_MODEL_PATH,
    ML_MODEL_PATHS, DNN_MODEL_PATH, DNN_MODEL_H5_PATH,
    DNN_SCALER_PATH, DNN_LABEL_ENCODER_PATH
)

@st.cache_resource(show_spinner=False)
def load_tfidf_vectorizer():
    """Loads and caches the Task 5 fitted TfidfVectorizer."""
    if os.path.exists(TFIDF_VECTORIZER_PATH):
        try:
            return joblib.load(TFIDF_VECTORIZER_PATH)
        except Exception as e:
            st.error(f"Error loading TF-IDF vectorizer: {e}")
            return None
    return None

@st.cache_resource(show_spinner=False)
def load_selected_feature_names():
    """Loads and caches the 100 selected feature names."""
    if os.path.exists(SELECTED_FEATURES_PATH):
        try:
            df = pd.read_csv(SELECTED_FEATURES_PATH)
            return df['Feature'].tolist()
        except Exception as e:
            st.error(f"Error loading selected features: {e}")
            return []
    return []

@st.cache_resource(show_spinner=False)
def load_ml_model(model_name: str):
    """Loads and caches a classical machine learning model."""
    path = ML_MODEL_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_dnn_artifacts():
    """Loads and caches the Task 7 Deep Neural Network model, scaler, and label encoder."""
    import tensorflow as tf
    dnn_model = None
    scaler = None
    label_encoder = None

    # Load Keras model
    for path in [DNN_MODEL_PATH, DNN_MODEL_H5_PATH]:
        if os.path.exists(path):
            try:
                dnn_model = tf.keras.models.load_model(path)
                break
            except Exception as e:
                print(f"Error loading DNN model from {path}: {e}")

    # Load Scaler
    if os.path.exists(DNN_SCALER_PATH):
        try:
            scaler = joblib.load(DNN_SCALER_PATH)
        except Exception as e:
            print(f"Error loading DNN scaler: {e}")

    # Load Label Encoder
    if os.path.exists(DNN_LABEL_ENCODER_PATH):
        try:
            label_encoder = joblib.load(DNN_LABEL_ENCODER_PATH)
        except Exception as e:
            print(f"Error loading DNN label encoder: {e}")

    return {
        "model": dnn_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "is_ready": dnn_model is not None and scaler is not None
    }

def get_model_readiness_status() -> Dict[str, Dict[str, Any]]:
    """Checks the availability and health of all model artifacts without raising errors."""
    status = {}
    for name, path in ML_MODEL_PATHS.items():
        exists = os.path.exists(path)
        status[name] = {
            "type": "Classical ML",
            "exists": exists,
            "path": str(path),
            "size_kb": round(os.path.getsize(path) / 1024, 1) if exists else 0,
            "ready": exists
        }

    dnn_exists = os.path.exists(DNN_MODEL_PATH) or os.path.exists(DNN_MODEL_H5_PATH)
    scaler_exists = os.path.exists(DNN_SCALER_PATH)
    status["Deep Neural Network (DNN)"] = {
        "type": "Deep Learning",
        "exists": dnn_exists and scaler_exists,
        "path": str(DNN_MODEL_PATH),
        "size_kb": round(os.path.getsize(DNN_MODEL_PATH) / 1024, 1) if os.path.exists(DNN_MODEL_PATH) else 0,
        "ready": dnn_exists and scaler_exists
    }
    return status
