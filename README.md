# Sentivise AI: Enterprise Product Review Sentiment Analysis & Intelligent Decision Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-brightgreen.svg)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed%20(100%25)-success.svg)](https://pytest.org/)

An enterprise-grade customer feedback intelligence and automated decision platform designed for e-commerce companies, product managers, customer experience (CX) teams, and business analysts. The platform bridges machine learning sentiment classification with symbolic rule-based expert system reasoning, transparent prioritization scoring, and Explainable AI (SHAP & LIME).

---

## 1. Project Overview & Objectives

In modern e-commerce, raw customer feedback contains critical operational signals regarding product defects, packaging compromises, courier transit delays, and customer support failures. Traditional systems stop at predicting a binary positive/negative sentiment label.

This system converts textual customer review telemetry into **actionable, auditable business decisions**:
1. **Analyzes customer reviews** using verified, pre-trained classical Machine Learning models (Logistic Regression, Random Forest, SVM, Decision Tree, XGBoost) and Deep Neural Networks (DNN).
2. **Classifies sentiment** with calibrated class probabilities and uncertainty boundaries.
3. **Identifies recurring product issues** categorized into five core operational themes (Product Quality, Logistics & Delivery, Packaging, Pricing & Value, Customer Support).
4. **Explains predictions** using local SHAP Shapley values, LIME surrogate models, and text Counterfactuals.
5. **Prioritizes business problems** via transparent multi-factor scoring (Frequency, Negative Ratio, Severity, Recency, Cross-Product Impact).
6. **Recommends departmental actions** using a dual-engine expert system (Forward and Backward Chaining).
7. **Maintains compliance** with local SQLite persistence and an immutable event audit trail.

---

## 2. System Architecture

```mermaid
graph TD
    A[Customer Reviews Input] --> B[Text Preprocessing & Sanitization]
    B --> C[Feature Engineering & Selection (Task 5)]
    C -->|12 Lexical/Structural + 88 TF-IDF N-grams| D[Model Inference Engine]
    
    subgraph Machine Learning & Deep Learning
        D --> M1[Logistic Regression Pipeline - Best 79.01%]
        D --> M2[Support Vector Machine - 78.85%]
        D --> M3[XGBoost Classifier - 78.55%]
        D --> M4[Random Forest Classifier - 75.66%]
        D --> M5[Decision Tree Classifier - 73.15%]
        D --> M6[Deep Neural Network (Task 7 DNN)]
    end
    
    D --> E[Calibrated Probability & Uncertainty Check]
    E --> F[Domain Aspect & Risk Heuristics]
    F --> G[Knowledge Base & Rule Engine]
    
    subgraph Decision Support & Reasoning
        G --> G1[Forward Chaining Engine (Data-Driven)]
        G --> G2[Backward Chaining Engine (Goal-Driven)]
        G --> G3[Issue Prioritization Scoring Formula]
    end
    
    G --> H[Intelligent Decision Center Workflow]
    H --> I[SQLite Persistence & Audit Trail]
    D --> J[Explainable AI: SHAP & LIME & Counterfactuals]
    H --> K[Automated Executive Briefings & Exports]
```

---

## 3. Project Directory Structure

```text
Ai-Review_System/
├── app.py                      # Main landing page, KPI telemetry & routing
├── config.py                   # Central configuration, paths, taxonomy, themes
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive documentation & setup guide
├── .gitignore                  # Git ignore rules for DB, caches, datasets
├── sentiment_system.db         # Persistent SQLite database (WAL mode)
│
├── pages/                      # 13 Dedicated Analytics & Business Modules
│   ├── 1_Executive_Dashboard.py
│   ├── 2_Review_Analyzer.py
│   ├── 3_Batch_Analytics.py
│   ├── 4_Customer_Insights.py
│   ├── 5_Decision_Center.py
│   ├── 6_Knowledge_Rules.py
│   ├── 7_ML_Evaluation.py
│   ├── 8_DNN_Model.py
│   ├── 9_Explainable_AI.py
│   ├── 10_Search_Lab.py
│   ├── 11_Data_Management.py
│   ├── 12_Reports_Audit.py
│   └── 13_Settings_About.py
│
├── components/                 # Standardized Light-Theme Design System
│   ├── ui_elements.py          # Custom CSS, KPI cards, badges, empty states
│   ├── charts.py               # Plotly donuts, stacked bars, confusion matrices
│   └── sidebar.py              # Navigation branding & system readiness badge
│
├── services/                   # High-Level Service Layer
│   ├── model_loader.py         # Streamlit cached loading of all models & scalers
│   ├── prediction_service.py   # Single inference, probabilities, uncertainty flag
│   ├── batch_service.py        # Chunked CSV batch inference, schema validation
│   ├── insights_service.py     # Aspect categorization, term frequencies, n-grams
│   ├── decision_service.py     # Issue triage, lifecycle management, note logging
│   ├── explainability_service.py # SHAP, LIME, counterfactuals, Task 8 summaries
│   ├── search_service.py       # BFS, DFS, A* execution and comparison
│   └── report_service.py       # JSON, CSV, and Markdown executive briefing exports
│
├── modules/                    # Core Mathematical & Algorithmic Modules
│   ├── preprocessing.py        # 12 engineered features + TF-IDF matrix slicing
│   ├── knowledge_base.py       # Formal definitions of Facts, Rules, Actions
│   ├── rule_engine.py          # Forward and Backward chaining engines + traces
│   ├── issue_prioritization.py # Transparent 5-factor mathematical priority formula
│   └── search_algorithms.py    # BFS, DFS, A* state space resolution algorithms
│
├── database/                   # Persistence Layer
│   ├── connection.py           # Thread-safe SQLite connection manager with WAL mode
│   ├── schema.py               # Table DDL (reviews, issues, notes, audit logs)
│   └── repository.py           # Parameterized CRUD operations & audit logger
│
├── models/                     # Organized Trained Models & Encoders
│   ├── ml/                     # Classical ML models (.pkl), vectorizer & feature list
│   └── dnn/                    # Deep Neural Network (.keras, .h5), scaler, label encoder
│
├── results/                    # Preserved Experimental Artifacts
│   ├── reports/                # Classification & evaluation CSV reports
│   └── figures/                # Confusion matrices, ROC curves, training curves
│
├── notebooks/                  # Jupyter Research Notebooks (Tasks 5–8)
│   ├── Task5_Feature_Engineering.ipynb
│   ├── Task6_Machine_Learning.ipynb
│   ├── Task7_Deep_Learning.ipynb
│   └── Task8_Explainable_AI.ipynb
│
├── scripts/                    # Generation & Reproduction Scripts
├── tests/                      # Automated Pytest Test Suite (30 Tests)
└── task8/                      # Preserved Task 8 XAI figures and metrics
```

---

## 4. Technology Stack & Dependencies

- **Language**: Python 3.11 / 3.12 / 3.13 (Verified on macOS Apple Silicon & Linux)
- **User Interface**: Streamlit (with custom CSS design system and Plotly integration)
- **Data Processing**: Pandas, NumPy, SciPy (Sparse matrices)
- **Machine Learning**: Scikit-Learn, Joblib, XGBoost
- **Deep Learning**: TensorFlow / Keras (Sequential DNN with Dropout and Sigmoid activation)
- **Explainable AI (XAI)**: SHAP (Linear & Additive Attribution), LIME (LimeTextExplainer)
- **Database & Persistence**: SQLite3 (Thread-safe, WAL mode, foreign keys enabled)
- **Testing**: Pytest (30 automated unit and integration tests)

---

## 4. Verified Label Mapping & Preprocessing

The ground-truth label mapping was verified directly from the dataset (`clean_product_reviews.csv`):
- `0`: **Negative Sentiment** (Original FastText tag: `__label__1`)
- `1`: **Positive Sentiment** (Original FastText tag: `__label__2`)

### Inference Preprocessing Pipeline:
1. **Input Sanitization**: Normalizes whitespace, strips non-printable tokens, and checks character bounds (3 to 5,000 characters).
2. **Lexical & Structural Features (12 features)**: `review_length`, `word_count`, `character_count`, `sentence_count`, `average_word_length`, `exclamation_count`, `question_count`, `uppercase_count`, `digit_count`, `positive_word_count`, `negative_word_count`, `sentiment_keyword_score`.
3. **TF-IDF Feature Extraction**: Unigrams and bigrams using the fitted `task5_tfidf_vectorizer.pkl`.
4. **Feature Selection**: Slices the sparse matrix down to the exact 100 features documented in `task5_selected_features.csv`.
5. **DNN Scaling**: For the deep learning model, transforms the 100 features into dense standard normal coordinates via `task7_scaler.pkl`.

---

## 5. Application Pages & Features

The platform is structured into 13 dedicated business intelligence and analytics modules:

| Page | Title | Key Functionality |
| :--- | :--- | :--- |
| **`app.py`** | **Platform Landing** | Executive system status, global KPIs, and quick navigation shortcuts. |
| **`pages/1_Executive_Dashboard.py`** | **Executive Dashboard** | Live KPIs, sentiment donut, feedback theme breakdown, and recent analysis feeds. |
| **`pages/2_Review_Analyzer.py`** | **Single Review Analyzer** | Real-time text analysis, calibrated confidence, local SHAP attributions, matched rules, and JSON/CSV export. |
| **`pages/3_Batch_Analytics.py`** | **Batch Review Analytics** | CSV upload, schema validation, duplicate detection, progress tracking, and appended CSV export. |
| **`pages/4_Customer_Insights.py`** | **Customer Insights** | Top positive/negative terms, 5-theme complaint breakdown, and repeated n-gram cluster detection. |
| **`pages/5_Decision_Center.py`** | **Intelligent Decision Center** | Core business workflow for issue triage, configurable priority scoring, and status transitions (Open, Investigating, Resolved, Dismissed). |
| **`pages/6_Knowledge_Rules.py`** | **Knowledge Base & Rule Engine** | 12 formal IF-THEN rules, forward-chaining trace, and backward-chaining goal proof tree. |
| **`pages/7_ML_Evaluation.py`** | **ML Model Evaluation** | Benchmark comparison of 5 classical classifiers, confusion matrices, and ROC curves. |
| **`pages/8_DNN_Model.py`** | **Deep Neural Network** | Architecture summary, Keras layers, training/validation loss curves, and classical vs DNN comparison. |
| **`pages/9_Explainable_AI.py`** | **Explainable AI (XAI) Lab** | Local SHAP attributions, perturbation-based LIME token weights, minimal Counterfactuals, and global importance. |
| **`pages/10_Search_Lab.py`** | **Search Algorithms Lab** | BFS, DFS, and A* Informed Search on an operational issue escalation workflow graph. |
| **`pages/11_Data_Management.py`** | **Data Management & Quality** | Schema inspector, missing values audit, label distribution, and import history logs. |
| **`pages/12_Reports_Audit.py`** | **Reports & Audit Trail** | Downloadable executive briefings in JSON & Markdown, plus compliance audit logs. |
| **`pages/13_Settings_About.py`** | **Settings & Responsible AI** | System architecture, artifact inventory, table purge controls, and ethical operating policies. |

---

## 6. Installation & Quick Start

### 1. Clone & Enter Repository
```bash
git clone <repo-url>
cd Ai-Review_System
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Running the Automated Test Suite

The project includes an automated test suite covering all modules, preprocessing routines, database operations, inference pipelines, and rule engines:

```bash
python3 -m pytest tests/ -v
```

### Test Suite Coverage:
- `tests/test_preprocessing.py`: Text cleaning, empty checks, 12 engineered features, 100-feature matrix construction.
- `tests/test_prediction.py`: Model loading, positive/negative predictions, probability bounds, error handling.
- `tests/test_rules.py`: Rule catalog integrity, condition evaluation, forward chaining, backward chaining proofs.
- `tests/test_prioritization.py`: Priority score formula, weight normalization, rationale generation.
- `tests/test_database.py`: SQLite initialization, record creation, issue lifecycle, note appending, audit trails.
- `tests/test_batch_analysis.py`: Batch schema validation, duplicate deduplication, appended output columns.
- `tests/test_search.py`: BFS, DFS, and A* goal achievement and optimality.
- `tests/test_explainability.py`: SHAP attributions, counterfactual generation, report formatting.

---

## 8. Batch CSV Input Format Specifications

When uploading batch customer review files in **Page 3 (Batch Review Analytics)**, the CSV should conform to the following schema:

| Column Name | Type | Required? | Example Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| `clean_review` / `review_text` | String | **Yes** | *"The headphones sound great, but the cord broke after 2 days."* | Raw customer feedback text. |
| `product_id` | String | Optional | `SKU-AUDIO-102` | Unique product identifier. |
| `product_name` | String | Optional | `Wireless Noise-Cancelling Headphones` | Human-readable product name. |
| `rating` | Float | Optional | `1.0` (Scale: 1.0 - 5.0) | Numerical customer satisfaction rating. |
| `date` | Date/String | Optional | `2026-10-05` | Review submission timestamp. |

---

## 9. Transparent Issue Prioritization Formula

The Intelligent Decision Center calculates priority scores using a documented mathematical formula:

$$\text{Priority Score} = w_f \cdot S_{\text{frequency}} + w_n \cdot S_{\text{neg\_ratio}} + w_s \cdot S_{\text{severity}} + w_r \cdot S_{\text{recency}} + w_p \cdot S_{\text{affected\_products}}$$

- $S_{\text{frequency}} = \min\left(100, \frac{\text{count}}{\text{benchmark\_max}} \times 100\right)$
- $S_{\text{neg\_ratio}} = \text{negative\_proportion} \times 100$
- $S_{\text{severity}} = 100 \text{ (Critical)}, 75 \text{ (High)}, 50 \text{ (Medium)}, 25 \text{ (Low)}$
- $S_{\text{recency}} = 100 \text{ (}\le 3\text{d)}, 80 \text{ (}\le 14\text{d)}, 60 \text{ (}\le 30\text{d)}, 40 \text{ (}\le 90\text{d)}, 20 \text{ (>90d)}$
- $S_{\text{affected\_products}} = \min\left(100, \frac{\text{distinct\_skus}}{\text{benchmark\_skus}} \times 100\right)$

Weights default to: $w_f = 0.25, w_n = 0.25, w_s = 0.20, w_r = 0.15, w_p = 0.15$ and can be customized directly in the UI.

---

## 10. Responsible AI & Operating Boundaries

1. **Assisted Decision Authority**: Recommendations are decision-support suggestions for human business analysts. The application does not automatically execute refunds, contact customers, or halt product lines without explicit human intervention.
2. **Confidence & Ambiguity**: Reviews whose model prediction probability falls within the uncertain range ($0.45 \le P(\text{Positive}) \le 0.55$) are flagged with an **Uncertainty Warning** requiring human review.
3. **Local Data Governance**: SQLite records are maintained locally. Customer data is not sent to external proprietary cloud LLM APIs.
4. **Auditability**: Every critical system event—including predictions, batch imports, issue status changes, and data purges—is recorded in the immutable audit log table.

---

## 11. Troubleshooting & FAQs

- **Port in use error**: If port 8501 is occupied, launch Streamlit on an alternate port:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **ModuleNotFoundError in pytest**: Run pytest as a Python module so the repository root is automatically added to `sys.path`:
  ```bash
  python3 -m pytest tests/ -v
  ```
- **DNN Model Keras warnings**: Safe compatibility warnings regarding NumPy scalar deprecations are suppressed by default in `config.py`.

---

## 12. Authors & Acknowledgments

- **Development Team**: Computational Intelligence Laboratory & AI/ML Engineering Team
- **Frameworks**: Streamlit, Scikit-Learn, TensorFlow, Plotly, SHAP, LIME
- **License**: MIT Academic / Enterprise License (2026)
