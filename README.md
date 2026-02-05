# Mental-Health-Illness-Detection-Using-Multi-Task-Learning
Symptom based mental illness detector for predicting five mental disorders

## Overview

This project implements a **production-oriented multitask mental health classification pipeline** designed to predict the presence of **five major psychiatric conditions** from structured clinical/survey data:

* PTSD
* Bipolar Disorder
* Schizophrenia
* Depression
* Anxiety Disorder

Unlike single-label classifiers, this system treats mental health assessment as a **multi-label decision-support problem**, where multiple disorders may co-exist. The pipeline emphasizes **robust preprocessing, model diversity, cross-validated evaluation, explainability, and reproducibility**, aligning with **industry-grade ML engineering standards**.

---

## Key Objectives

* Handle **severe class imbalance** using SMOTE
* Compare **classical ML, deep learning, and tabular DL models** under identical evaluation protocols
* Perform **5-fold cross-validation** for statistically reliable results
* Provide **per-disorder ROC-AUC**, not just accuracy
* Enable **model explainability** using SHAP and LIME
* Produce artifacts suitable for **clinical decision-support**, not diagnosis

---

## System Architecture

```
Raw Clinical Dataset (CSV)
        ↓
Column Normalization & Cleaning
        ↓
Per-Label SMOTE Balancing
        ↓
Feature Scaling (StandardScaler)
        ↓
Multi-Model Training (5-Fold CV)
 ┌───────────────────────────────────────┐
 │ Decision Tree                         │
 │ Random Forest                         │
 │ Gradient Boosting                     │
 │ Extra Trees                           │
 │ K-Nearest Neighbors                   │
 │ Support Vector Machine                │
 │ TabNet (Deep Tabular Learning)        │
 │ Keras Multitask Neural Network        │
 └───────────────────────────────────────┘
        ↓
Per-Disorder Metrics + ROC-AUC
        ↓
Explainability (SHAP + LIME)
```

---

## Dataset Description

* Input: `Mentalillness.csv`
* Format: Structured tabular data
* Labels (multi-label):

  * PTSD
  * Bipolar disorder
  * Schizophrenia
  * Depression
  * Anxiety disorder

### Class Imbalance (Before Processing)

The original dataset exhibits **significant class imbalance**, which is typical in real-world mental health datasets.

### SMOTE Strategy (Per-Label)

Instead of applying SMOTE globally, the pipeline:

* Applies **SMOTE independently for each disorder**
* Merges balanced datasets via feature-level outer joins
* Preserves multi-label structure

This avoids bias toward dominant disorders while maintaining realism.

---

## Models Implemented

### Classical Machine Learning

* Decision Tree
* Random Forest
* Gradient Boosting
* Extra Trees
* K-Nearest Neighbors
* Support Vector Machine (probabilistic)

All classical models are trained using a **one-vs-rest strategy** for each disorder.

### Deep Learning Models

#### 1. TabNet (PyTorch)

* Attention-based feature selection
* Sparse decision steps
* Strong performance on structured data
* GPU-aware training

#### 2. Multitask Neural Network (Keras)

* Shared feature representation
* Multiple sigmoid outputs
* Binary cross-entropy loss
* End-to-end multitask optimization

---

## Evaluation Protocol

* **5-Fold Cross-Validation** (KFold, shuffled)
* Metrics reported per disorder:

  * Precision
  * Recall
  * F1-Score
  * ROC-AUC
* Final metrics computed on **aggregated predictions across folds**

This avoids optimistic bias and improves statistical reliability.

---

## Performance Summary (High-Level)

| Model             | Strengths               | Limitations                  |
| ----------------- | ----------------------- | ---------------------------- |
| Decision Tree     | Fast, interpretable     | Overfitting risk             |
| Random Forest     | Excellent AUC, robust   | Higher inference cost        |
| Gradient Boosting | Strong balance          | Longer training              |
| Extra Trees       | High variance reduction | Less interpretable           |
| KNN               | Simple baseline         | Poor recall for rare classes |
| SVM               | Best overall balance    | Computationally expensive    |
| TabNet            | Feature-aware DL        | Training complexity          |
| Multitask NN      | Shared learning         | Requires tuning              |

**Best Overall AUC:** SVM / Random Forest / TabNet
**Best Recall Balance:** TabNet & Multitask Neural Network

---

## Explainability & Trust

### SHAP (Global Interpretability)

* Identifies globally influential features
* Measures feature contribution magnitude
* Suitable for clinical audit trails

### LIME (Local Interpretability)

* Explains individual predictions
* Useful for clinician-level inspection

Explainability is treated as a **first-class system component**, not an afterthought. For only one disorder, it is shown for sample (as we want to publish this work).

---

## Output Artifacts

* Per-fold CSV reports (`results_fold*.csv`)
* Final AUC summaries (`final_auc_scores.csv`)
* Normalized confusion matrices (TIFF, publication-quality)
* Explainability plots (SHAP & LIME)

All artifacts are reproducible and version-controlled.

---

## Engineering Considerations

* Modular pipeline design
* Deterministic random seeds
* GPU-aware execution (TabNet)
* Scalable to additional disorders
* Ready for REST inference wrapping (Spring Boot / FastAPI)

---

## Future Extensions

* Cost-sensitive loss functions
* Temporal modeling (longitudinal data)
* Federated learning for privacy
* Clinical rule integration
* Deployment as a secured web service

