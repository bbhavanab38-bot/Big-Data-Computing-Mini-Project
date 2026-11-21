"""Phase 3 - Apache Spark: Distributed data processing on YARN or local cluster.

Performs:
1. Distributed data loading from HDFS (with automatic local fallback)
2. Distributed aggregations (Geography x Gender churn rates)
3. Distributed feature extraction (behavioral ratios & activity scores)
4. Distributed customer segmentation (PySpark MLlib K-Means clustering)
5. Segment-level churn analysis
6. Optimized Parquet storage for downstream ML pipelines
7. Detailed execution metrics logging (duration, partitions, row counts)
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
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler, VectorAssembler

from src import config

MASTER = os.environ.get("SPARK_MASTER", "local[*]")
INPUT = os.environ.get("BANK_INPUT", "/banking_data/processed_bank_data.csv")
OUTPUT = os.environ.get("BANK_OUTPUT", "/banking_data/output")
WRITE_MODE = "overwrite" if os.environ.get("OVERWRITE", "1").strip().lower() in ("1", "true", "yes", "overwrite") else "errorifexists"


def get_spark() -> SparkSession:
    """Build and return an optimized SparkSession for Apple Silicon / YARN."""
    builder = (
        SparkSession.builder
        .master(MASTER)
        .appName("BankingRetention-SparkPreprocessing")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.driver.memory", "2g")
    )
    return builder.getOrCreate()


def resolve_input_path(spark: SparkSession, desired_path: str) -> str:
    """Check if HDFS path is accessible; if not, fall back to local project path."""
    try:
        # Test read 1 row
        spark.read.csv(desired_path, header=True).take(1)
        return desired_path
    except Exception as exc:
        print(f"  [Spark Info] Desired input {desired_path} not directly accessible ({exc}); using local file fallback.")
        local_csv = str(config.PROCESSED_FILE)
        return local_csv


def resolve_output_path(spark: SparkSession, desired_path: str) -> str:
    """Check if HDFS output is writable; fall back to local reports/spark staging if offline."""
    if desired_path.startswith("/banking_data"):
        try:
            spark.read.csv(INPUT, header=True).take(1)
            return desired_path
        except Exception:
            fallback = str(ROOT / "reports" / "spark" / "staging_output")
            return fallback
    return desired_path


def main():
    total_start = time.time()
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    metrics = {
        "spark_version": spark.version,
        "master": MASTER,
        "stages": {},
    }

    # 1. READ ------------------------------------------------------------------------
    t0 = time.time()
    effective_input = resolve_input_path(spark, INPUT)
    effective_output = resolve_output_path(spark, OUTPUT)
    print(f"\n[Spark Ingestion] Reading from: {effective_input}")

    df = spark.read.csv(effective_input, header=True, inferSchema=True)
    num_rows = df.count()
    num_cols = len(df.columns)
    num_parts = df.rdd.getNumPartitions()
    metrics["stages"]["1_read"] = {
        "duration_sec": round(time.time() - t0, 3),
        "input_path": effective_input,
        "row_count": num_rows,
        "column_count": num_cols,
        "partitions": num_parts,
    }
    print(f"  Rows: {num_rows} | Columns: {num_cols} | Partitions: {num_parts} (time: {metrics['stages']['1_read']['duration_sec']}s)")

    # 2. AGGREGATION -----------------------------------------------------------------
    t0 = time.time()
    print("\n[Spark Aggregation] Computing Demographic & Geographic Churn Aggregations...")
    agg = (df.groupBy("Geography", "Gender")
             .agg(F.count("*").alias("Customers"),
                  F.round(F.avg("Balance"), 2).alias("AvgBalance"),
                  F.round(F.avg("CreditScore"), 1).alias("AvgCreditScore"),
                  F.round(F.avg("TransactionCount"), 1).alias("AvgTransactions"),
                  F.round(F.avg("Exited") * 100, 2).alias("ChurnRatePct"))
             .orderBy(F.desc("ChurnRatePct")))
    agg.show()
    agg_out = f"{effective_output}/aggregation_geo_gender"
    agg.write.mode(WRITE_MODE).option("header", True).csv(agg_out)
    metrics["stages"]["2_aggregation"] = {
        "duration_sec": round(time.time() - t0, 3),
        "output_path": agg_out,
    }

    # 3. FEATURE EXTRACTION ----------------------------------------------------------
    t0 = time.time()
    print("\n[Spark Feature Engineering] Calculating Domain Ratios & Behavioral Scores...")
    df = (df.withColumn("TransactionFrequency", F.col("TransactionCount") / (F.col("Tenure") + 1))
            .withColumn("AccountAgeScore", F.least(F.col("Tenure") / F.greatest(F.col("Age") - 18, F.lit(1)), F.lit(1.0)))
            .withColumn("LoanToBalanceRatio", F.least(F.col("LoanAmount") / (F.col("Balance") + 1), F.lit(50.0)))
            .withColumn("DigitalUsageScore", 0.7 * F.least(F.col("OnlineBankingUsage") / 100, F.lit(1.0)) + 0.3 * F.col("HasCrCard"))
            .withColumn("CustomerActivityScore", 0.4 * F.col("IsActiveMember")
                        + 0.3 * F.least(F.col("TransactionCount") / 50, F.lit(1.0))
                        + 0.3 * F.least(F.col("NumOfProducts") / 4, F.lit(1.0))))
    metrics["stages"]["3_feature_extraction"] = {
        "duration_sec": round(time.time() - t0, 3),
        "engineered_features": ["TransactionFrequency", "AccountAgeScore", "LoanToBalanceRatio", "DigitalUsageScore", "CustomerActivityScore"],
    }

    # 4. CUSTOMER SEGMENTATION (K-Means) -----------------------------------------------
    t0 = time.time()
    print("\n[Spark MLlib Segmentation] Running Distributed K-Means Clustering (k=4)...")
    seg_cols = ["Age", "Balance", "CreditScore", "TransactionCount", "EstimatedSalary", "CustomerActivityScore"]
    seg_pipe = Pipeline(stages=[
        VectorAssembler(inputCols=seg_cols, outputCol="seg_raw"),
        StandardScaler(inputCol="seg_raw", outputCol="seg_scaled", withMean=True, withStd=True),
        KMeans(k=4, seed=42, featuresCol="seg_scaled", predictionCol="Segment"),
    ])
    df = seg_pipe.fit(df).transform(df).drop("seg_raw", "seg_scaled")
    metrics["stages"]["4_kmeans_clustering"] = {
        "duration_sec": round(time.time() - t0, 3),
        "clusters": 4,
        "feature_dimensions": len(seg_cols),
    }

    # 5. CHURN ANALYSIS ----------------------------------------------------------------
    t0 = time.time()
    print("\n[Spark Segment Churn Analysis] Aggregating Churn Hazard Across Clusters...")
    seg_summary = (df.groupBy("Segment")
                     .agg(F.count("*").alias("Customers"),
                          F.round(F.avg("Age"), 1).alias("AvgAge"),
                          F.round(F.avg("Balance"), 0).alias("AvgBalance"),
                          F.round(F.avg("CustomerActivityScore"), 3).alias("AvgActivity"),
                          F.round(F.avg("Exited") * 100, 2).alias("ChurnRatePct"))
                     .orderBy("Segment"))
    seg_summary.show()
    seg_out = f"{effective_output}/segment_churn_summary"
    seg_summary.write.mode(WRITE_MODE).option("header", True).csv(seg_out)
    metrics["stages"]["5_segment_summary"] = {
        "duration_sec": round(time.time() - t0, 3),
        "output_path": seg_out,
    }

    # 6. PERSIST ENGINEERED DATASET (Parquet) -----------------------------------------
    t0 = time.time()
    parquet_out = f"{effective_output}/features_parquet"
    print(f"\n[Spark Persistence] Writing engineered feature store to Parquet: {parquet_out}...")
    df.write.mode(WRITE_MODE).parquet(parquet_out)
    metrics["stages"]["6_parquet_persistence"] = {
        "duration_sec": round(time.time() - t0, 3),
        "parquet_output": parquet_out,
        "final_columns": len(df.columns),
    }

    total_time = round(time.time() - total_start, 3)
    metrics["total_execution_sec"] = total_time
    print(f"\n Spark Preprocessing Completed Successfully in {total_time}s")

    # Save metrics JSON to local reports directory
    metrics_path = ROOT / "reports" / "spark" / "spark_preprocessing_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"  Execution metrics saved to: {metrics_path}")

    spark.stop()


if __name__ == "__main__":
    main()
