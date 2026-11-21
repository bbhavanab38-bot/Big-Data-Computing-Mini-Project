@echo off
REM Phase 2 - create HDFS folder and upload cleaned data (HDFS + YARN must be running)
hdfs dfs -mkdir -p /banking_data
hdfs dfs -put -f data\processed\processed_bank_data.csv /banking_data
hdfs dfs -ls /banking_data
