#!/usr/bin/env zsh
# Full pipeline runner for macOS / Apple Silicon
SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

echo "=== 1. Python Data & Preprocessing ==="
python main.py --step data
python main.py --step preprocess

echo "=== 2. HDFS Setup ==="
zsh scripts/hdfs_setup.sh

echo "=== 3. Spark on YARN: Preprocessing & Training ==="
zsh scripts/spark_submit.sh spark/spark_preprocessing.py
zsh scripts/spark_submit.sh spark/spark_training.py
zsh scripts/spark_submit.sh spark/spark_validate_all_models.py

echo "=== 4. Export Spark Results ==="
zsh scripts/export_spark_results.sh

echo "=== 5. Python Modeling & Evaluation ==="
python main.py --step eda
python main.py --step train
python main.py --step evaluate
python main.py --step powerbi
python main.py --step predict

