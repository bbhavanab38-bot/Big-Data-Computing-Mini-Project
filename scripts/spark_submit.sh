#!/usr/bin/env zsh
# Spark submit on YARN or local cluster for macOS
SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [ -z "$1" ]; then
    echo "Usage: scripts/spark_submit.sh <spark_script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift

MASTER="${SPARK_MASTER:-yarn}"

echo "Submitting ${SCRIPT} to Spark (master: ${MASTER})..."
spark-submit \
    --master "${MASTER}" \
    --deploy-mode client \
    --driver-memory 2g \
    --executor-memory 2g \
    --num-executors 2 \
    "${SCRIPT}" "$@"

