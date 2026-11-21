#!/usr/bin/env bash
# Environment configuration for Hadoop, Spark, and Java on macOS (Apple Silicon)

# Detect script directory portably across bash and zsh
if [ -n "$ZSH_VERSION" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%N}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prioritize Java 17 / 21 as required by the stack
if [ -d "/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home" ]; then
    export JAVA_HOME="/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home"
elif [ -d "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home" ]; then
    export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
elif [ -n "$(/usr/libexec/java_home -v 17 2>/dev/null)" ]; then
    export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null)"
elif [ -n "$(/usr/libexec/java_home 2>/dev/null)" ]; then
    export JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null)"
fi

export PATH="$JAVA_HOME/bin:$PATH"

# Hadoop installation paths (Homebrew default on Apple Silicon)
export HADOOP_HOME="${HADOOP_HOME:-/opt/homebrew/opt/hadoop/libexec}"
export HADOOP_CONF_DIR="$HADOOP_HOME/etc/hadoop"
export YARN_CONF_DIR="$HADOOP_CONF_DIR"
export PATH="$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH"

# Python and Virtualenv
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3 || which python)"
fi

export PYSPARK_PYTHON="$PYTHON_BIN"
export PYSPARK_DRIVER_PYTHON="$PYTHON_BIN"

# Spark Home from PySpark or standalone
if [ -z "$SPARK_HOME" ]; then
    export SPARK_HOME="$("$PYTHON_BIN" -c 'import pyspark, os; print(os.path.dirname(pyspark.__file__))' 2>/dev/null)"
fi
if [ -n "$SPARK_HOME" ]; then
    export PATH="${PROJECT_DIR}/.venv/bin:$SPARK_HOME/bin:$PATH"
fi
