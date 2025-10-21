from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder \
    .appName("TrafficViolationIngestion") \
    .master("local[*]") \
    .getOrCreate()

traffic_schema = StructType([
    StructField("Violation_ID", StringType(), False),
    StructField("Timestamp", StringType(), True),  
    StructField("Latitude", DoubleType(), True),
    StructField("Longitude", DoubleType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity", IntegerType(), True)
])

df = spark.read.option("header", True).schema(traffic_schema).csv("traffic.csv")

df.show(5)

spark.stop()
