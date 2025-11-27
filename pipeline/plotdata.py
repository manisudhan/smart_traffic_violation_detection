# Week 3–4: Traffic Violation Analysis + Visualization + PDF Report
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# Step 1: Initialize Spark
spark = SparkSession.builder \
    .appName("TrafficViolationAnalysis_Week3_4") \
    .master("local[*]") \
    .getOrCreate()

print("Spark session started successfully")

# Step 2: Load cleaned dataset
df = spark.read.parquet("cleaned_traffic_data.parquet")
print("Cleaned dataset loaded")

# Step 3: Derive time-based features
df_time = df.withColumn("hour", F.hour("Timestamp")) \
            .withColumn("day_of_week", F.date_format("Timestamp", "E")) \
            .withColumn("month", F.month("Timestamp")) \
            .withColumn("year", F.year("Timestamp"))

print(" Derived time-based features (hour, day_of_week, month, year)")

# Step 4: Aggregations
violations_per_hour = df_time.groupBy("hour").agg(F.count("*").alias("total_violations")).orderBy("hour")
violations_per_day = df_time.groupBy("day_of_week").agg(F.count("*").alias("total_violations")).orderBy("day_of_week")
violations_by_type = df_time.groupBy("Violation_Type").agg(F.count("*").alias("total_violations")).orderBy(F.desc("total_violations"))
violations_per_location = df_time.groupBy("Latitude", "Longitude").agg(F.count("*").alias("total_violations")).orderBy(F.desc("total_violations"))
top_locations = violations_per_location.limit(10)

print("Aggregations completed for time, type, and location-based analysis")

# Step 5: Save outputs as Parquet (Week 3)
violations_per_hour.write.mode("overwrite").parquet("output/time_based/hourly")
violations_per_day.write.mode("overwrite").parquet("output/time_based/daily")
violations_by_type.write.mode("overwrite").parquet("output/offense_type/type_summary")
violations_per_location.write.mode("overwrite").parquet("output/location_based/all_locations")
top_locations.write.mode("overwrite").parquet("output/location_based/top_locations")

print(" Aggregated results saved as Parquet tables")

# Step 6: Convert to Pandas for visualization (Week 4)
pdf_hour = violations_per_hour.toPandas()
pdf_day = violations_per_day.toPandas()
pdf_type = violations_by_type.toPandas()
pdf_toploc = top_locations.toPandas()

# Step 7: Visualizations + PDF Export
plt.style.use('seaborn-v0_8-darkgrid')

# Create a PDF to save all plots
pdf_report = PdfPages("Traffic_Violation_Report.pdf")

# --- Cover Page ---
fig_cover, ax_cover = plt.subplots(figsize=(8, 6))
ax_cover.axis('off')
ax_cover.text(0.5, 0.8, "🚦 TRAFFIC VIOLATION ANALYSIS REPORT", fontsize=16, ha='center', weight='bold')
ax_cover.text(0.5, 0.65, "Weeks 3–4 Project", fontsize=12, ha='center')
ax_cover.text(0.5, 0.45, "Data Source: cleaned_traffic_data.parquet", fontsize=10, ha='center')
ax_cover.text(0.5, 0.35, f"Total Records: {df.count()}", fontsize=10, ha='center')
date_range = df.agg(F.min("Timestamp"), F.max("Timestamp")).collect()[0]
ax_cover.text(0.5, 0.25, f"Date Range: {date_range[0]}  →  {date_range[1]}", fontsize=9, ha='center')
ax_cover.text(0.5, 0.1, "Generated automatically using PySpark + Matplotlib", fontsize=8, ha='center', style='italic')
pdf_report.savefig(fig_cover)
plt.close(fig_cover)

# --- Violations by Hour ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(pdf_hour["hour"], pdf_hour["total_violations"], color='steelblue')
ax1.set_title("🚗 Violations per Hour of Day")
ax1.set_xlabel("Hour of Day (0–23)")
ax1.set_ylabel("Total Violations")
ax1.set_xticks(range(0, 24))
plt.tight_layout()
pdf_report.savefig(fig1)
plt.close(fig1)

# --- Violations by Day of Week ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(pdf_day["day_of_week"], pdf_day["total_violations"], color='coral')
ax2.set_title(" Violations per Day of Week")
ax2.set_xlabel("Day of Week")
ax2.set_ylabel("Total Violations")
plt.tight_layout()
pdf_report.savefig(fig2)
plt.close(fig2)

# --- Violations by Type ---
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.barh(pdf_type["Violation_Type"], pdf_type["total_violations"], color='mediumseagreen')
ax3.set_title("🚦 Violations by Type of Offense")
ax3.set_xlabel("Total Violations")
ax3.set_ylabel("Violation Type")
plt.tight_layout()
pdf_report.savefig(fig3)
plt.close(fig3)

# --- Top 10 Locations ---
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.barh(range(len(pdf_toploc)), pdf_toploc["total_violations"], color='slateblue')
ax4.set_yticks(range(len(pdf_toploc)))
ax4.set_yticklabels([f"({lat:.2f}, {lon:.2f})" for lat, lon in zip(pdf_toploc["Latitude"], pdf_toploc["Longitude"])])
ax4.set_title(" Top 10 Violation Hotspots")
ax4.set_xlabel("Total Violations")
ax4.set_ylabel("Latitude, Longitude")
plt.tight_layout()
pdf_report.savefig(fig4)
plt.close(fig4)

# --- Summary Tables ---
fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
ax_summary.axis('off')
summary_text = (
    " Top 5 Violation Types:\n\n" +
    pdf_type.head().to_string(index=False) +
    "\n\n Top 10 Violation Hotspots:\n\n" +
    pdf_toploc.to_string(index=False)
)
ax_summary.text(0.01, 0.99, summary_text, ha='left', va='top', fontsize=9, family='monospace')
pdf_report.savefig(fig_summary)
plt.close(fig_summary)

# Close PDF
pdf_report.close()

print(" All visualizations and summaries saved to Traffic_Violation_Report.pdf")

# -------------------------------------------------
# Step 8: Summary Preview on Console
# -------------------------------------------------
print("\n Violations by Type (Top 5):")
print(pdf_type.head())

print("\n Top 10 Violation Hotspots:")
print(pdf_toploc)

# -------------------------------------------------
# Step 9: Stop Spark
# -------------------------------------------------
spark.stop()
print("Spark session stopped — Week 3–4 visualization + PDF report completed successfully!")
