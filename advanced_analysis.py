# -------------------------------------------------
# Milestone 3: Weeks 5–6 — Advanced Pattern Analysis & Hotspot Identification
# -------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans

# Optional: Geo support (grid visualization)
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEO_SUPPORT = True
except ImportError:
    GEO_SUPPORT = False
    print("⚠️ GeoPandas not installed — skipping grid-based grouping")

# Step 1: Initialize Spark
spark = SparkSession.builder \
    .appName("TrafficViolation_AdvancedAnalysis") \
    .master("local[*]") \
    .getOrCreate()

print("✅ Spark session started")

# Step 2: Load cleaned dataset
df = spark.read.parquet("cleaned_traffic_data.parquet")

# Ensure proper timestamp conversion
df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))

record_count = df.count()
print(f"✅ Dataset loaded: {record_count} records")

# Step 3: Add time-derived features
df = df.withColumn("hour", F.hour("Timestamp")) \
       .withColumn("weekday", F.date_format("Timestamp", "E")) \
       .withColumn("day_type",
                   F.when(F.col("weekday").isin(["Sat", "Sun"]), "Weekend").otherwise("Weekday")) \
       .withColumn("hour_window", (F.col("hour") / 3).cast("int") * 3)

print("✅ Derived 3-hour windows and weekday/weekend split")

# Step 4: Time-based analysis
violations_by_window = df.groupBy("hour_window").agg(F.count("*").alias("total_violations")).orderBy("hour_window")
violations_by_daytype = df.groupBy("day_type").agg(F.count("*").alias("total_violations"))
violations_type_time = df.groupBy("Violation_Type", "hour_window").agg(F.count("*").alias("total_violations"))

print("✅ Time-based grouping and correlation by violation type done")

# Step 5: Spatial grouping (grid cells)
if "Latitude" in df.columns and "Longitude" in df.columns:
    df_geo = df.withColumn("lat_bin", (F.col("Latitude") * 10).cast("int")) \
               .withColumn("lon_bin", (F.col("Longitude") * 10).cast("int")) \
               .withColumn("grid_cell", F.concat_ws("_", F.col("lat_bin"), F.col("lon_bin")))

    grid_stats = df_geo.groupBy("grid_cell").agg(
        F.count("*").alias("total_violations"),
        F.avg("Latitude").alias("avg_lat"),
        F.avg("Longitude").alias("avg_lon")
    ).orderBy(F.desc("total_violations"))

    top_hotspots = grid_stats.limit(10)
    print("✅ Spatial grouping (grid-based) completed")
else:
    print("⚠️ Latitude/Longitude not found — skipping spatial grouping")

# -------------------------------------------------
# Step 6: Optional — K-Means clustering for hotspots
# -------------------------------------------------
if "Latitude" in df.columns and "Longitude" in df.columns:
    pdf_geo = df.select("Latitude", "Longitude").dropna().toPandas()
    if len(pdf_geo) > 20:  # need enough data points
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        pdf_geo["cluster"] = kmeans.fit_predict(pdf_geo[["Latitude", "Longitude"]])
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=["Latitude", "Longitude"])
        print("✅ K-Means clustering completed — 5 hotspots identified")
    else:
        pdf_geo["cluster"] = -1
        centers = pd.DataFrame()
        print("⚠️ Not enough geo points for clustering")
else:
    pdf_geo = pd.DataFrame()
    centers = pd.DataFrame()

# Step 7: Convert results to Pandas for visualization
pdf_window = violations_by_window.toPandas()
pdf_daytype = violations_by_daytype.toPandas()
pdf_type_time = violations_type_time.toPandas()
pdf_hotspot = top_hotspots.toPandas() if "top_hotspots" in locals() else pd.DataFrame()

# Step 8: Visualization & PDF Report
# -------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign issue
plt.rcParams["font.family"] = "DejaVu Sans"  # Emoji-safe fallback
plt.style.use("seaborn-v0_8-darkgrid")

pdf = PdfPages("Traffic_Advanced_Analysis_Report.pdf")

# --- Violations by 3-hour window ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(pdf_window["hour_window"], pdf_window["total_violations"], width=2.5, color="royalblue")
ax1.set_title("Violations per 3-Hour Window")
ax1.set_xlabel("Start Hour of Window")
ax1.set_ylabel("Total Violations")
pdf.savefig(fig1)
plt.close(fig1)

# --- Weekday vs Weekend ---
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(pdf_daytype["day_type"], pdf_daytype["total_violations"], color="coral")
ax2.set_title("Violations: Weekday vs Weekend")
ax2.set_xlabel("Day Type")
ax2.set_ylabel("Total Violations")
pdf.savefig(fig2)
plt.close(fig2)

# --- Violation Type vs Time (FIXED HEATMAP) ---
fig3, ax3 = plt.subplots(figsize=(10, 6))
pivot_data = pdf_type_time.pivot(
    index="Violation_Type",
    columns="hour_window",
    values="total_violations"
).fillna(0)
sns.heatmap(pivot_data, cmap="YlGnBu", ax=ax3)
ax3.set_title("Violation Types vs Time of Day (3-Hour Windows)")
ax3.set_xlabel("Hour Window Start")
ax3.set_ylabel("Violation Type")
pdf.savefig(fig3)
plt.close(fig3)

# --- Top Hotspot Grid Cells ---
if not pdf_hotspot.empty:
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    ax4.scatter(pdf_hotspot["avg_lon"], pdf_hotspot["avg_lat"],
                s=pdf_hotspot["total_violations"] * 5, alpha=0.6, color="crimson")
    ax4.set_title("Top 10 Hotspot Grid Cells")
    ax4.set_xlabel("Longitude")
    ax4.set_ylabel("Latitude")
    pdf.savefig(fig4)
    plt.close(fig4)

# --- K-Means Clusters (if available) ---
if not pdf_geo.empty and not centers.empty:
    fig5, ax5 = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=pdf_geo, x="Longitude", y="Latitude", hue="cluster",
                    palette="tab10", s=15, ax=ax5)
    ax5.scatter(centers["Longitude"], centers["Latitude"], c="black", s=100,
                marker="X", label="Cluster Centers")
    ax5.set_title("K-Means Violation Hotspot Clusters (5 Zones)")
    ax5.legend()
    pdf.savefig(fig5)
    plt.close(fig5)

# --- Summary Page ---
fig_summary, ax_summary = plt.subplots(figsize=(8, 5))
ax_summary.axis("off")
summary_text = (
    "📊 ADVANCED TRAFFIC VIOLATION ANALYSIS\n\n"
    f"Records analyzed: {record_count}\n\n"
    "Key Highlights:\n"
    "- Time grouping in 3-hour windows\n"
    "- Weekday vs Weekend comparison\n"
    "- Violation-type vs Time heatmap\n"
    "- Spatial grid hotspot detection\n"
    "- (Optional) K-Means clustering on coordinates\n\n"
    "Top Hotspot Zones:\n"
    f"{pdf_hotspot.head().to_string(index=False) if not pdf_hotspot.empty else 'N/A'}"
)
ax_summary.text(0.02, 0.98, summary_text, ha="left", va="top", fontsize=9, family="monospace")
pdf.savefig(fig_summary)
plt.close(fig_summary)

pdf.close()
print("✅ Advanced analysis report saved → Traffic_Advanced_Analysis_Report.pdf")

# Step 9: Save Outputs
violations_by_window.write.mode("overwrite").parquet("output/advanced/time_window")
violations_by_daytype.write.mode("overwrite").parquet("output/advanced/daytype")
violations_type_time.write.mode("overwrite").parquet("output/advanced/type_time")
if "grid_stats" in locals():
    grid_stats.write.mode("overwrite").parquet("output/advanced/spatial_grid")

print("✅ Outputs saved as Parquet files")

# Step 10: Stop Spark
spark.stop()
print("🏁 Spark session stopped — Milestone 3 completed successfully")
