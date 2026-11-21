#!/usr/bin/env bash
# Check live status of Hadoop HDFS, YARN, and PySpark
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

echo "=========================================================="
echo "          BIG DATA PLATFORM STATUS REPORT                 "
echo "=========================================================="

echo "\n--- 1. Java Virtual Machine Processes (jps) ---"
jps

echo "\n--- 2. HDFS Cluster Health & Capacity ---"
"${HADOOP_HOME}/bin/hdfs" dfsadmin -report 2>&1 | head -n 35

echo "\n--- 3. HDFS Banking Directory (/banking_data) ---"
"${HADOOP_HOME}/bin/hdfs" dfs -ls /banking_data 2>&1 || echo "Directory /banking_data not found or HDFS offline."

echo "\n--- 4. YARN Node Manager Nodes ---"
"${HADOOP_HOME}/bin/yarn" node -list 2>&1 || echo "YARN ResourceManager offline."

echo "\n--- 5. Python & PySpark Environment ---"
echo "Python Interpreter: $PYSPARK_PYTHON"
"$PYSPARK_PYTHON" -c "import pyspark; print('PySpark version:', pyspark.__version__)" 2>&1 || echo "PySpark import failed."

echo "\n=========================================================="
