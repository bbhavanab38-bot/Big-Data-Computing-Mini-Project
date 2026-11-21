#!/usr/bin/env bash
# Start Hadoop HDFS and YARN daemons directly without requiring macOS SSH Remote Login
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

echo "=== Starting Hadoop HDFS Services (macOS Apple Silicon) ==="
"${HADOOP_HOME}/bin/hdfs" --daemon start namenode
"${HADOOP_HOME}/bin/hdfs" --daemon start datanode
"${HADOOP_HOME}/bin/hdfs" --daemon start secondarynamenode

echo "Leaving Safe Mode if engaged..."
sleep 2
"${HADOOP_HOME}/bin/hdfs" dfsadmin -safemode leave 2>/dev/null || true

echo "=== Starting Hadoop YARN Services ==="
"${HADOOP_HOME}/bin/yarn" --daemon start resourcemanager
"${HADOOP_HOME}/bin/yarn" --daemon start nodemanager

echo "=== Active Java Processes (jps) ==="
jps
echo "Hadoop daemons started successfully."
