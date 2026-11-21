@echo off
REM Full run: Python pipeline -> HDFS -> Spark on YARN
python main.py --step data
python main.py --step preprocess
call scripts\hdfs_setup.bat
call scripts\spark_submit.bat spark\spark_preprocessing.py
call scripts\spark_submit.bat spark\spark_training.py
python main.py --step eda
python main.py --step train
python main.py --step evaluate
python main.py --step powerbi
python main.py --step predict
