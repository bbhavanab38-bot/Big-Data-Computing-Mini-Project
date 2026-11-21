#!/usr/bin/env zsh
# Export Spark results from HDFS /banking_data/output to reports/spark/
SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${PROJECT_DIR}/reports/spark"
mkdir -p "${TARGET_DIR}"

HDFS_OUTPUT="/banking_data/output"
DATASETS=(
    "aggregation_geo_gender"
    "segment_churn_summary"
    "spark_model_comparison"
    "spark_model_comparison_validated"
    "spark_model_comparison_validated_all"
    "spark_model_comparison_weighted"
    "customer_segments"
)

echo "Exporting Spark outputs from HDFS ${HDFS_OUTPUT} to ${TARGET_DIR}..."

for item in "${DATASETS[@]}"; do
    if hdfs dfs -test -e "${HDFS_OUTPUT}/${item}" 2>/dev/null; then
        echo "Fetching ${item}..."
        rm -rf "${TARGET_DIR}/${item}"
        hdfs dfs -get "${HDFS_OUTPUT}/${item}" "${TARGET_DIR}/"
    else
        echo "Notice: ${HDFS_OUTPUT}/${item} not found on HDFS (may not have been generated yet)."
    fi
done

echo "Current local contents of ${TARGET_DIR}:"
ls -la "${TARGET_DIR}"

