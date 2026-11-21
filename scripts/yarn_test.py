if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("BankingYARNTest").getOrCreate()

    print("Spark version:", spark.version)
    print("YARN test result:", spark.range(10).count())

    spark.stop()

