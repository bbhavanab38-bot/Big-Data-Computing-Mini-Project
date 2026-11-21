"""Validated churn-class metrics (Exited=1 positive) for ALL four Spark MLlib models.
Writes ONLY to /banking_data/output/spark_model_comparison_validated_all (new path).
"""
import os
import sys

# Guarantee matching Python interpreter across Spark driver and executors
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import (DecisionTreeClassifier, GBTClassifier,
                                       LogisticRegression, RandomForestClassifier)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

MASTER = os.environ.get("SPARK_MASTER", "yarn")
OUTPUT = os.environ.get("BANK_OUTPUT", "/banking_data/output")
RESULT_PATH = f"{OUTPUT}/spark_model_comparison_validated_all"

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "TransactionCount",
           "OnlineBankingUsage", "CustomerComplaints", "LoanAmount", "HasCrCard", "IsActiveMember",
           "TransactionFrequency", "AccountAgeScore", "LoanToBalanceRatio", "DigitalUsageScore", "CustomerActivityScore"]
CATEGORICAL = ["Geography", "Gender", "LoanStatus"]


def main():
    spark = SparkSession.builder.master(MASTER).appName("BankChurn-ValidateAllModels").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(f"{OUTPUT}/features_parquet").withColumn("label", F.col("Exited").cast("double"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)        # same split as spark_training.py
    pos = train.filter("label = 1").count(); neg = train.count() - pos
    train = train.withColumn("w", F.when(F.col("label") == 1, neg / max(pos, 1)).otherwise(1.0)).cache()
    print(f"Train rows: {train.count()}  Test rows: {test.count()}")

    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CATEGORICAL]
    encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in CATEGORICAL], outputCols=[f"{c}_ohe" for c in CATEGORICAL])
    assembler = VectorAssembler(inputCols=NUMERIC + [f"{c}_ohe" for c in CATEGORICAL], outputCol="features")
    prep = indexers + [encoder, assembler]

    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", weightCol="w", maxIter=100),
        "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=6),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, maxDepth=10, seed=42),
        "Gradient Boosting": GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=4, seed=42),
    }
    binary = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                    metricName="recallByLabel", metricLabel=1.0)
    rows = []
    for name, clf in models.items():
        pred = Pipeline(stages=prep + [clf]).fit(train).transform(test).cache()
        c = pred.agg(
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("TP"),
            F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("FP"),
            F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("TN"),
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("FN"),
        ).first()
        tp, fp, tn, fn = (int(c[k] or 0) for k in ("TP", "FP", "TN", "FN"))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        rbl = recall_eval.evaluate(pred)
        print(f"{name}: TN={tn} FP={fp} FN={fn} TP={tp} | recall manual={recall:.4f} spark={rbl:.4f}")
        rows.append((name, "Exited=1 (Churned)", tp, fp, tn, fn, precision, recall, f1, accuracy,
                     binary.setMetricName("areaUnderROC").evaluate(pred),
                     binary.setMetricName("areaUnderPR").evaluate(pred),
                     rbl, abs(recall - rbl) < 1e-12))
        pred.unpersist()

    cols = ["Model", "PositiveClass", "TP", "FP", "TN", "FN", "ChurnPrecision", "ChurnRecall", "ChurnF1",
            "Accuracy", "ROC_AUC", "PR_AUC", "RecallByLabel", "RecallMatches"]
    result = spark.createDataFrame(rows, cols).orderBy(F.desc("ROC_AUC"))
    result.show(truncate=False)
    result.coalesce(1).write.mode("overwrite").option("header", True).csv(RESULT_PATH)
    print(f"Saved to {RESULT_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
