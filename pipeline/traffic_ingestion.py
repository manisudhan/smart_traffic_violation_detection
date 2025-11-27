from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from pyspark.sql.functions import col, when, trim, lower, to_timestamp

# --------------------------------------------
# Step 1: Create Spark session
# --------------------------------------------
spark = SparkSession.builder \
    .appName("TrafficViolationIngestion") \
    .master("local[*]") \
    .getOrCreate()

print("Spark Session created successfully")

# --------------------------------------------
# Step 2: Define schema for the CSV
# --------------------------------------------
traffic_schema = StructType([
    StructField("Violation_ID", StringType(), False),
    StructField("Timestamp", StringType(), True),  
    StructField("Latitude", DoubleType(), True),
    StructField("Longitude", DoubleType(), True),
    StructField("Violation_Type", StringType(), True),
    StructField("Vehicle_Type", StringType(), True),
    StructField("Severity", IntegerType(), True)
])

# --------------------------------------------
# Step 3: Read CSV data
# --------------------------------------------
df = spark.read.option("header", True).schema(traffic_schema).csv("traffic.csv")

print(" Data successfully read from CSV")
df.show(10)

# --------------------------------------------
# Step 4: Handle missing or null values
# --------------------------------------------
# Drop rows where critical fields are missing
df_cleaned = df.na.drop(subset=["Violation_ID", "Timestamp", "Violation_Type"])

# Replace missing latitude/longitude with default or mean values
df_cleaned = df_cleaned.fillna({
    "Latitude": 0.0,
    "Longitude": 0.0,
    "Severity": 1
})

print(" Missing values handled")

# --------------------------------------------
# Step 5: Standardize timestamps
# --------------------------------------------
df_cleaned = df_cleaned.withColumn(
    "Timestamp", 
    to_timestamp(col("Timestamp"), "yyyy-MM-dd HH:mm:ss")
)

# Remove rows where timestamp conversion failed (became null)
df_cleaned = df_cleaned.filter(col("Timestamp").isNotNull())

print(" Timestamps standardized")

# --------------------------------------------
# Step 6: Standardize categorical fields
# --------------------------------------------
df_cleaned = df_cleaned.withColumn("Violation_Type", trim(lower(col("Violation_Type"))))
df_cleaned = df_cleaned.withColumn("Vehicle_Type", trim(lower(col("Vehicle_Type"))))

# Step 7: Validate Violation Types
valid_violations = ["speeding", "red light", "illegal parking", "no helmet"]

df_cleaned = df_cleaned.withColumn(
    "Violation_Type",
    when(col("Violation_Type").isin(valid_violations), col("Violation_Type")).otherwise("unknown")
)

print("Violation types validated")


df_cleaned.write.mode("overwrite").parquet("cleaned_traffic_data.parquet")

print(" Cleaned data saved in Parquet format")


print("Total records before cleaning:", df.count())
print("Total records after cleaning:", df_cleaned.count())

df_cleaned.show(10)

spark.stop()
print(" Spark session stopped. Week 2 milestone completed successfully!")
