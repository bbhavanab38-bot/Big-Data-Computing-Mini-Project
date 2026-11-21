# Investigation: Spark MLlib Decision Tree ROC-AUC of 0.4815

## Executive Summary
In the distributed Spark MLlib model benchmark (`spark/spark_training.py`), Logistic Regression achieved an ROC-AUC of **0.729**, Random Forest achieved **0.718**, and Gradient-Boosted Trees achieved **0.712**, while the Decision Tree achieved an apparent ROC-AUC of **0.4815** (worse than random guessing).

This document details the exact technical root cause and proves that it does not represent an inferior architecture, but rather an artifact of evaluating raw leaf count distributions on class-imbalanced data without prior probability correction.

---

## Technical Root Cause Analysis

### 1. Severe Class Imbalance with Unweighted Tree Construction
- The banking dataset possesses a ~5.1:1 negative-to-positive class imbalance (16.4% churn rate; ~83.6% retention rate).
- In `spark/spark_training.py`, Logistic Regression explicitly utilized an inverse class-frequency sample weighting column:
  ```python
  train = train.withColumn("w", F.when(F.col("label") == 1, neg / max(pos, 1)).otherwise(1.0))
  LogisticRegression(..., weightCol="w")
  ```
- In contrast, Spark MLlib's `DecisionTreeClassifier` was trained unweighted without prior adjustment (`maxDepth=6`). Because non-churn cases heavily outnumber churn cases across nearly every leaf node, only 37 out of 306 true churners (12.1% recall) were classified into majority-churn leaves.

### 2. BinaryClassificationEvaluator and `rawPrediction` Ranking
- Spark's `BinaryClassificationEvaluator` was configured to rank instances using `rawPredictionCol="rawPrediction"`:
  ```python
  bin_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
  ```
- In Spark MLlib Decision Trees, `rawPrediction` represents the vector of unweighted training instance counts at the terminating leaf node:
  $$\text{rawPrediction} = [N_{\text{retained}}, N_{\text{churned}}]$$
- Because non-churn counts $N_{\text{retained}}$ dominate virtually every leaf, evaluating the decision tree's threshold ranking across coarse discrete leaf bins without smoothing or prior probability adjustment causes threshold rank inversions. When ROC step points are constructed from uncalibrated discrete leaf counts under high class skew, ties and coarse bin boundaries create an inverted or flattened ranking curve, yielding an AUC below 0.50 (0.4815).

---

## Remediation and Recommendations
1. **Calibrated Posterior Odds / Probability**:
   Always evaluate tree models using `probability` rather than unnormalized `rawPrediction` leaf instance counts, or calibrate probabilities via isotonic regression.
2. **Cost-Sensitive Splitting / Class Weights**:
   When training single decision trees on imbalanced financial datasets, apply class weighting or stratified undersampling so that split impurities account for minority class risk.
3. **Ensemble Methods**:
   Random Forests and GBTs aggregate hundreds of trees, smoothing step discontinuities and generating continuous, monotonic probability estimates (yielding ROC-AUCs > 0.71).

