"""Phase 3/6 - Distributed model training with Spark MLlib on YARN or local cluster.

Models: Logistic Regression, Decision Tree, Random Forest, Gradient-Boosted Trees
(XGBoost is evaluated in scikit-learn in src/train_model.py; Spark ML ships no native XGBoost).
"""
import json
import os
import sys
import time
from pathlib import Path

# Guarantee matching Python interpreter across Spark driver and executors
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

MASTER = os.environ.get("SPARK_MASTER", "local[*]")
OUTPUT = os.environ.get("BANK_OUTPUT", "/banking_data/output")
WRITE_MODE = "overwrite" if os.environ.get("OVERWRITE", "1").strip().lower() in ("1", "true", "yes", "overwrite") else "errorifexists"

NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary", "TransactionCount",
           "OnlineBankingUsage", "CustomerComplaints", "LoanAmount", "HasCrCard", "IsActiveMember",
           "TransactionFrequency", "AccountAgeScore", "LoanToBalanceRatio", "DigitalUsageScore", "CustomerActivityScore"]
CATEGORICAL = ["Geography", "Gender", "LoanStatus"]


def resolve_features_path(spark: SparkSession, desired_path: str) -> str:
    """Check if HDFS Parquet features are accessible; fall back to local staging if offline."""
    try:
        spark.read.parquet(desired_path).take(1)
        return desired_path
    except Exception as exc:
        print(f"  [Spark Info] Parquet path {desired_path} not directly accessible ({exc}); checking local staging.")
        local_staging = str(ROOT / "reports" / "spark" / "staging_output" / "features_parquet")
        return local_staging


def main():
    start_time = time.time()
    spark = (
        SparkSession.builder
        .master(MASTER)
        .appName("BankChurnPrediction-Training")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    metrics_log = {
        "spark_version": spark.version,
        "master": MASTER,
        "models": {},
    }

    parquet_path = f"{OUTPUT}/features_parquet"
    effective_parquet = resolve_features_path(spark, parquet_path)
    print(f"\n[Spark Training] Reading Parquet features from: {effective_parquet}")

    df = spark.read.parquet(effective_parquet).withColumn("label", F.col("Exited").cast("double"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    train.cache()

    train_count = train.count()
    test_count = test.count()
    print(f"  Train split: {train_count} rows | Test split: {test_count} rows")

    # Class-imbalance weight for Logistic Regression
    pos = train.filter("label = 1").count()
    neg = train_count - pos
    train = train.withColumn("w", F.when(F.col("label") == 1, neg / max(pos, 1)).otherwise(1.0))

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
    bin_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    accuracy_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
    recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0)

    rows, fitted = [], {}
    for name, clf in models.items():
        t0 = time.time()
        model = Pipeline(stages=prep + [clf]).fit(train)
        pred = model.transform(test)
        fitted[name] = model

        acc = float(accuracy_eval.evaluate(pred))
        prec = float(precision_eval.evaluate(pred))
        rec = float(recall_eval.evaluate(pred))
        f1 = float(f1_eval.evaluate(pred))
        auc = float(bin_eval.setMetricName("areaUnderROC").evaluate(pred))
        train_sec = round(time.time() - t0, 3)

        rows.append((name, acc, prec, rec, f1, auc))
        metrics_log["models"][name] = {
            "training_sec": train_sec,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
        }
        print(f"  Trained {name:20s} in {train_sec:6.2f}s | ROC-AUC: {auc:.4f} | Accuracy: {acc*100:.2f}%")

    result = spark.createDataFrame(rows, ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]).orderBy(F.desc("ROC_AUC"))
    print("\n=== SPARK MLlib MODEL COMPARISON ===")
    result.show(truncate=False)

    cmp_out = f"{OUTPUT}/spark_model_comparison"
    result.coalesce(1).write.mode(WRITE_MODE).option("header", True).csv(cmp_out)

    best = result.first()["Model"]
    model_writer = fitted[best].write()
    if WRITE_MODE == "overwrite":
        model_writer = model_writer.overwrite()
    best_out = f"{OUTPUT}/best_spark_model"
    model_writer.save(best_out)
    print(f"\nBest Spark Model: '{best}' saved to {best_out}")

    metrics_log["best_model"] = best
    metrics_log["total_training_sec"] = round(time.time() - start_time, 3)

    metrics_path = ROOT / "reports" / "spark" / "spark_training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as mf:
        json.dump(metrics_log, mf, indent=2)
    print(f"  Training metrics saved to: {metrics_path}")

    spark.stop()


if __name__ == "__main__":
    main()
