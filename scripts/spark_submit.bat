@echo off
REM Usage: scripts\spark_submit.bat spark\spark_preprocessing.py
REM Client deploy mode is the reliable choice for Windows single-node YARN.
if "%PYSPARK_PYTHON%"=="" set PYSPARK_PYTHON=python
spark-submit --master yarn --deploy-mode client --driver-memory 2g --executor-memory 2g --num-executors 2 %1
