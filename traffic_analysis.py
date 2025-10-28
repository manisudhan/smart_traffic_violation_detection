from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------------------------
# Step 1: Initialize Spark
# -------------------------------------------------
spark = SparkSession.builder \
    .appName("TrafficViolationAnalysis") \
    .master("local[*]") \
    .getOrCreate()

print("✅ Spark session started for analysis")

# -------------------------------------------------
# Step 2: Read cleaned Parquet data
# -------------------------------------------------
df = spark.read.parquet("cleaned_traffic_data.parquet")
print("✅ Cleaned Parquet file loaded")

# -------------------------------------------------
# Step 3: Derive time-based features
# -------------------------------------------------
df_time = df.withColumn("hour", F.hour("Timestamp")) \
            .withColumn("day_of_week", F.date_format("Timestamp", "E")) \
            .withColumn("month", F.month("Timestamp")) \
            .withColumn("year", F.year("Timestamp"))

# -------------------------------------------------
# Step 4: Aggregations
# -------------------------------------------------

# 4.1 Violations per hour
violations_per_hour = df_time.groupBy("hour").agg(F.count("*").alias("total_violations")).orderBy("hour")

# 4.2 Violations per day of week
violations_per_day = df_time.groupBy("day_of_week").agg(F.count("*").alias("total_violations")).orderBy("day_of_week")

# 4.3 Violations by type
violations_by_type = df_time.groupBy("Violation_Type").agg(F.count("*").alias("total_violations")).orderBy(F.desc("total_violations"))

# 4.4 Cross-tab: Violation type × Hour of day
cross_tab = df_time.crosstab("Violation_Type", "hour")

# -------------------------------------------------
# Step 5: Location-based analysis
# -------------------------------------------------
violations_per_location = df_time.groupBy("Latitude", "Longitude") \
                                 .agg(F.count("*").alias("total_violations")) \
                                 .orderBy(F.desc("total_violations"))

top_locations = violations_per_location.limit(10)

# -------------------------------------------------
# Step 6: Save outputs as Parquet
# -------------------------------------------------
violations_per_hour.write.mode("overwrite").parquet("output/time_based/hourly")
violations_per_day.write.mode("overwrite").parquet("output/time_based/daily")
violations_by_type.write.mode("overwrite").parquet("output/offense_type/type_summary")
cross_tab.write.mode("overwrite").parquet("output/offense_type/type_hour_matrix")
violations_per_location.write.mode("overwrite").parquet("output/location_based/all_locations")
top_locations.write.mode("overwrite").parquet("output/location_based/top_locations")

print("✅ Aggregated results saved as Parquet tables")

# -------------------------------------------------
# Step 7: Quick Preview
# -------------------------------------------------
print("📊 Violations by type:")
violations_by_type.show(5)

print("📊 Top 10 Locations:")
top_locations.show(10)

spark.stop()
print("✅ Spark session stopped — Week 3–4 milestone complete!")
