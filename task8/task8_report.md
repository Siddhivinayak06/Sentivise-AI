# Task 8 – Explainable AI (XAI) for Product Review Sentiment Analysis

**Computational Intelligence Laboratory**  
**AI-Based Product Review Sentiment Analysis and Intelligent Decision Support System**  
**Author:** AI/ML Engineering & XAI Team  
**Date:** October 2026  

---

## 1. Title
**Task 8: Explainable AI (XAI) for Product Review Sentiment Classification**

---

## 2. Aim
To design, implement, evaluate, and critically compare Explainable AI (XAI) methodologies—specifically **SHAP (SHapley Additive exPlanations)**, **LIME (Local Interpretable Model-agnostic Explanations)**, and **Counterfactual Explanations**—to make machine learning predictions transparent, auditable, and interpretable for business intelligence and decision support.

---

## 3. Objectives
1. Inspect and reuse existing datasets, preprocessing pipelines, and trained classification models from Tasks 5–7.
2. Verify and document target label mappings (`0: Negative`, `1: Positive`).
3. Select representative test instances spanning positive, negative, mixed/ambiguous, and edge-case sentiments.
4. Establish baseline sentiment predictions with calibrated probabilities.
5. Implement local and global feature attribution using **SHAP**.
6. Implement local perturbative surrogate explanations using **LIME**.
7. Develop a constrained text-based **Counterfactual explanation generator** to identify minimal prediction-flipping edits.
8. Compare explanation consistency, Jaccard overlap, and runtime efficiency across methods.
9. Discuss Responsible AI, algorithmic fairness, and ethical deployment considerations.
10. Persist all generated figures, metrics, and tabular reports.

---

## 4. Dataset and Model Details

### 4.1 Dataset Summary
- **Source Dataset:** `clean_product_reviews.csv` (399,996 total reviews)
- **Training Set (`X_train.csv`):** 319,996 reviews
- **Testing Set (`X_test.csv`):** 80,000 reviews
- **Target Column:** `sentiment_numeric`
- **Verified Label Mapping:**
  - `0` $\rightarrow$ **Negative** (199,997 instances in full dataset)
  - `1` $\rightarrow$ **Positive** (199,999 instances in full dataset)

### 4.2 Feature Representation
- **Selected Features (Task 5):** 100 features documented in `task5_selected_features.csv`
  - 12 engineered features (`sentiment_keyword_score`, `positive_word_count`, `negative_word_count`, `word_count`, `review_length`, `average_word_length`, punctuation frequencies)
  - 88 TF-IDF unigram and bigram features (`great`, `waste`, `bad`, `love`, `money`, `not`, `worst`, etc.)

### 4.3 Evaluated Model
- **Primary Model:** Logistic Regression Pipeline (`best_sentiment_model.pkl`) trained in Task 6, comprising a `MaxAbsScaler` followed by `LogisticRegression(max_iter=1000)`.
- **Baseline Accuracy:** 79.01% on unseen test data; ROC-AUC: 0.9066.

---

## 5. SHAP Implementation

### 5.1 Theoretical Background
SHAP computes Shapley values from cooperative game theory, attributing credit to each feature $i$ by computing its marginal contribution across all possible feature subsets:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S))$$

SHAP provides mathematical guarantees of **Local Accuracy**, **Missingness**, and **Consistency**.

### 5.2 Implementation Details
- **Explainer:** `shap.LinearExplainer` instantiated with a 100-instance empirical background dataset extracted from `X_train`.
- **Base Value ($E[f(x)]$):** -0.0076 (balanced prior expectation).
- **Outputs Generated:**
  - `task8/graphs/01_shap_positive_explanation.png`
  - `task8/graphs/02_shap_negative_explanation.png`
  - `task8/graphs/03_shap_global_importance.png`
  - `task8/results/task8_shap_results.csv`

### 5.3 Key Findings
- In positive reviews (`REV_01_POS`), `sentiment_keyword_score` (+0.435) and `positive_word_count` (+0.328) provide the largest positive attribution.
- In negative reviews (`REV_02_NEG`), `sentiment_keyword_score` (-0.435) and negative keyword tokens (`waste`, `bad`, `terrible`) drive attribution strongly below zero.
- The global summary plot confirms that polar lexical indicators dominate model decisions across the corpus.

---

## 6. LIME Implementation

### 6.1 Theoretical Background
LIME trains an interpretable linear surrogate model $g$ locally around review $x$ by minimizing surrogate loss weighted by exponential proximity kernel $\pi_x(z)$:

$$\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)$$

### 6.2 Implementation Details
- **Explainer:** `LimeTextExplainer(class_names=['Negative', 'Positive'], random_state=42)`
- **Prediction Callback:** `lime_predict_proba` accepting arbitrary perturbed strings and executing the full end-to-end feature extraction and pipeline inference.
- **Perturbation Sample Size:** 500 samples per instance.
- **Outputs Generated:**
  - `task8/graphs/04_lime_positive_explanation.png`
  - `task8/graphs/05_lime_negative_explanation.png`
  - `task8/results/task8_lime_results.csv`

### 6.3 Key Findings
- LIME token-level attributions clearly highlight individual words (`amazing`, `great`, `love` for positive; `terrible`, `broke`, `waste` for negative).
- LIME is model-agnostic and directly inspects words in context, making it immediately understandable to non-technical stakeholders.

---

## 7. Counterfactual Explanations

### 7.1 Concept and Optimization
A counterfactual explanation finds the minimal input modification $x'$ such that the model prediction changes class:

$$\arg\min_{x'} d(x, x') \quad \text{subject to} \quad f(x') \neq f(x)$$

### 7.2 Implementation Details
- Developed an antonym-substitution search strategy constrained by edit distance.
- Evaluated candidates through the full inference pipeline and ranked by minimal changes made.
- Outputs generated: `task8/graphs/07_counterfactual_comparison.png` and `task8/results/task8_counterfactual_results.csv`.

### 7.3 Measured Results
| Review ID | Category | Original Prediction (Confidence) | Counterfactual Prediction (Confidence) | Edits Made | Substitutions |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `REV_01_POS` | Clearly Positive | Positive (0.9979) | Negative (0.5042) | 1 | `'amazing' -> 'disappointing'` |
| `REV_02_NEG` | Clearly Negative | Negative (1.0000) | Negative (1.0000) | 0 | Strong negative accumulation (robust) |
| `REV_03_AMB` | Ambiguous / Mixed | Negative (0.9997) | Positive (0.8924) | 3 | `'good' -> 'bad', 'poor' -> 'nice', 'disappointing' -> 'amazing'` |

---

## 8. Results and Visualizations

All 8 figures have been generated and saved to `task8/graphs/`:

1. **`01_shap_positive_explanation.png`**: Local SHAP feature bar chart for positive sentiment.
2. **`02_shap_negative_explanation.png`**: Local SHAP feature bar chart for negative sentiment.
3. **`03_shap_global_importance.png`**: Top 15 global features ranked by mean absolute SHAP value.
4. **`04_lime_positive_explanation.png`**: Token-level surrogate weights for positive review.
5. **`05_lime_negative_explanation.png`**: Token-level surrogate weights for negative review.
6. **`06_shap_lime_comparison.png`**: Side-by-side attribution comparison between SHAP and LIME.
7. **`07_counterfactual_comparison.png`**: Prediction confidence shifts before and after counterfactual intervention.
8. **`08_explanation_runtime_comparison.png`**: Log-scale runtime comparison across methods.

---

## 9. Comparison of Explanation Methods

| Criterion | SHAP | LIME | Counterfactual |
| :--- | :--- | :--- | :--- |
| **Primary Objective** | Additive Shapley feature attribution | Local linear surrogate approximation | Minimal input change to flip prediction |
| **Mathematical Basis** | Axiomatic cooperative game theory | Local perturbation loss minimization | Constrained search over edit distance |
| **Scope** | Local instance & Global cohort | Local instance only | Local instance only |
| **Output Representation** | Numerical feature attributions | Token importance weights | Modified natural language string |
| **Model Agnostic** | Requires model-specific explainer | Fully model-agnostic | Fully model-agnostic |
| **Execution Speed** | Ultra-fast (0.00027s per review) | Moderate (0.1309s per review) | Fast (0.0113s per review) |
| **Primary Limitation** | Relies on feature matrix structure | Perturbation sampling instability | Fluency & syntax constraints |

---

## 10. Explanation Consistency (SHAP vs. LIME)

Evaluating the top features identified by SHAP against the top tokens identified by LIME:
- **Positive Review Jaccard Similarity:** `0.1111` (Overlap: `love`)
- **Negative Review Jaccard Similarity:** `0.2500` (Overlap: `waste`, `bad`)

### Why Discrepancies Occur:
1. **Feature Space vs. Token Space:** SHAP operates in the 100-dimensional engineered feature space (which includes structural metrics like `sentiment_keyword_score` and `positive_word_count`), whereas LIME operates directly on raw text tokens.
2. **Sampling Variability:** LIME uses random local perturbations, while SHAP computes exact closed-form linear Shapley attributions.

---

## 11. Responsible AI Considerations
1. **Correlation vs. Causation:** Explanations highlight tokens correlated with sentiment; they do not establish causal customer sentiment drivers.
2. **Human-in-the-Loop:** Automated decision systems must require human approval before triggering consequential actions (e.g., supplier penalties or product recalls).
3. **Linguistic Blindspots:** Sarcasm (*"Best way to throw money away"*) can trick both model predictions and keyword-level explanations.
4. **Data Privacy:** Customer reviews may contain PII (names, dates); explainability engines must mask sensitive tokens.

---

## 12. Limitations
1. **Surrogate Fidelity:** LIME's local linear surrogate is an approximation of the decision boundary and may exhibit variance across runs.
2. **Synthetic Counterfactuals:** Word substitutions may produce grammatically awkward sentences if not paired with a language model.
3. **Static Background Set:** SHAP linear background distribution assumes feature independence among correlated n-grams.

---

## 13. Conclusion
Task 8 establishes an end-to-end Explainable AI framework for product review sentiment classification:
- **SHAP** provides theoretically rigorous, additive global and local attributions.
- **LIME** provides intuitive, instance-level word importance for non-technical users.
- **Counterfactuals** reveal actionable decision thresholds by discovering minimal prediction-flipping edits.
- All code, figures, benchmarks, and deliverables are fully implemented, verified, and ready for deployment.
