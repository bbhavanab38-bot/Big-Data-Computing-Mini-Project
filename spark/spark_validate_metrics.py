"""Validate the saved best Spark model using churn (Exited=1) as the positive class."""
import os

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession, functions as F


MASTER = os.environ.get("SPARK_MASTER", "yarn")
OUTPUT = os.environ.get("BANK_OUTPUT", "/banking_data/output")
FEATURES_PATH = f"{OUTPUT}/features_parquet"
MODEL_PATH = f"{OUTPUT}/best_spark_model"
RESULT_PATH = f"{OUTPUT}/spark_model_comparison_validated"
WRITE_MODE = "overwrite" if os.environ.get("OVERWRITE", "").strip().lower() in ("1", "true", "yes", "overwrite") else "errorifexists"


def main():
    spark = (SparkSession.builder.master(MASTER)
             .appName("BankChurnPrediction-ValidatedMetrics")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        data = spark.read.parquet(FEATURES_PATH).withColumn("label", F.col("Exited").cast("double"))
        _, test = data.randomSplit([0.8, 0.2], seed=42)
        model = PipelineModel.load(MODEL_PATH)
        predictions = model.transform(test).cache()

        counts = predictions.agg(
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("TP"),
            F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("FP"),
            F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("TN"),
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("FN"),
        ).first()

        tp, fp, tn, fn = (int(counts[name] or 0) for name in ("TP", "FP", "TN", "FN"))
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / total if total else 0.0

        recall_eval = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0
        )
        binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
        recall_by_label = recall_eval.evaluate(predictions)
        roc_auc = binary_eval.setMetricName("areaUnderROC").evaluate(predictions)
        pr_auc = binary_eval.setMetricName("areaUnderPR").evaluate(predictions)
        recall_matches = abs(recall - recall_by_label) < 1e-12

        print("Confusion matrix (actual rows: Exited=0, Exited=1; predicted columns: 0, 1):")
        print("                 Predicted 0  Predicted 1")
        print(f"Actual Exited=0  {tn:12d}  {fp:12d}")
        print(f"Actual Exited=1  {fn:12d}  {tp:12d}")
        print(f"Churn precision: {precision:.6f}")
        print(f"Churn recall:    {recall:.6f}")
        print(f"RecallByLabel:   {recall_by_label:.6f} (metricLabel=1.0)")
        print(f"Recall matches:  {recall_matches}")
        print(f"Churn F1:        {f1:.6f}")
        print(f"Accuracy:        {accuracy:.6f}")
        print(f"ROC-AUC:         {roc_auc:.6f}")
        print(f"PR-AUC:          {pr_auc:.6f}")

        result = spark.createDataFrame([(
            "Saved best Spark MLlib model", "Exited=1 (Churned)", tp, fp, tn, fn,
            precision, recall, f1, accuracy, roc_auc, pr_auc, recall_by_label,
            recall_matches,
        )], [
            "Model", "PositiveClass", "TP", "FP", "TN", "FN", "ChurnPrecision",
            "ChurnRecall", "ChurnF1", "Accuracy", "ROC_AUC", "PR_AUC",
            "RecallByLabel", "RecallMatches",
        ])
        result.coalesce(1).write.mode(WRITE_MODE).option("header", True).csv(RESULT_PATH)
        print(f"Validated metrics written to {RESULT_PATH}")
        predictions.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
