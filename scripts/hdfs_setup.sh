#!/usr/bin/env zsh
# Set up HDFS folder and upload processed bank data
SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "Creating HDFS directory /banking_data..."
hdfs dfs -mkdir -p /banking_data

echo "Uploading processed_bank_data.csv to /banking_data..."
hdfs dfs -put -f "${PROJECT_DIR}/data/processed/processed_bank_data.csv" /banking_data

echo "Listing /banking_data on HDFS:"
hdfs dfs -ls /banking_data

