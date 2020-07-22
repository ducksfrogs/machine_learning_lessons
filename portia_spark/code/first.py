from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Basics").getOrCreate()
