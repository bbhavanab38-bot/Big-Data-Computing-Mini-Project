#!/usr/bin/env bash
# Stop Hadoop HDFS and YARN daemons cleanly
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

echo "=== Stopping Hadoop YARN Services ==="
"${HADOOP_HOME}/bin/yarn" --daemon stop nodemanager
"${HADOOP_HOME}/bin/yarn" --daemon stop resourcemanager

echo "=== Stopping Hadoop HDFS Services ==="
"${HADOOP_HOME}/bin/hdfs" --daemon stop secondarynamenode
"${HADOOP_HOME}/bin/hdfs" --daemon stop datanode
"${HADOOP_HOME}/bin/hdfs" --daemon stop namenode

echo "=== Active Java Processes (jps) ==="
jps
echo "Hadoop daemons stopped cleanly."
