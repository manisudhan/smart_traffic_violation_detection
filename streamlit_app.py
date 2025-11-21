"""
Streamlit dashboard for Traffic Violation Analysis
Reads precomputed parquet outputs created by a PySpark pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# Map display: use pydeck if available
try:
    import pydeck as pdk
    USE_PYDECK = True
except ImportError:
    USE_PYDECK = False

# Page Configuration
st.set_page_config(
    page_title="🚨 Traffic Violation Dashboard | Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Utility Functions

@st.cache_data(ttl=3600)
def load_parquet(path):
    """
    Load parquet into pandas safely.
    Handles both single files and partitioned directories from Spark.
    Returns DataFrame or None if missing.
    """
    if not os.path.exists(path):
        st.warning(f"Data path not found: {path}")
        return None
    
    try:
        if os.path.isdir(path):
            # Path is a directory, likely from Spark (partitioned)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
            if not files:
                st.error(f"Directory {path} contains no .parquet files.")
                return None
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            return df
        else:
            # Path is a single file
            return pd.read_parquet(path)
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

def convert_df_to_bytes(df, fmt="csv"):
    """Utility to convert DataFrame to bytes for downloading."""
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    else:
        # Default to JSON
        return df.to_json(orient="records", date_format="iso").encode("utf-8")

@st.cache_data(ttl=3600)
def load_all_data():
    """Loads all required and optional dataframes."""
    with st.spinner("⏳ Loading PySpark data outputs..."):
        data = {
            "hour": load_parquet("output/time_based/hourly"),
            "day": load_parquet("output/time_based/daily"),
            "type": load_parquet("output/offense_type/type_summary"),
            "top_loc": load_parquet("output/location_based/top_locations"),
            "all_loc": load_parquet("output/location_based/all_locations"),
            "grid": load_parquet("output/advanced/spatial_grid"),
            "type_time": load_parquet("output/advanced/type_time"),
        }
    return data

def setup_sidebar(df_type, df_all_loc):
    """Creates the sidebar filters and returns selected values."""
    st.sidebar.header("🗄️ Filter Options")
    
    # 1. Violation Type filter
    if df_type is not None:
        viol_types = sorted(df_type["Violation_Type"].unique())
        selected_types = st.sidebar.multiselect(
            "Select Violation Type(s)", options=viol_types, default=viol_types
        )
    else:
        st.sidebar.info("Violation type data not loaded.")
        selected_types = []

    # 2. Date range filter
    selected_date_range = None
    if df_all_loc is not None and "Timestamp" in df_all_loc.columns:
        df_all_loc["Timestamp"] = pd.to_datetime(df_all_loc["Timestamp"], errors="coerce")
        min_date = df_all_loc["Timestamp"].min()
        max_date = df_all_loc["Timestamp"].max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            selected_date_range = st.sidebar.date_input(
                "Filter by Date Range", 
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
        else:
            st.sidebar.info("Could not parse timestamps for date filter.")
    
    # 3. Severity filter
    selected_severity = None
    if df_all_loc is not None and "Severity" in df_all_loc.columns:
        severity_vals = sorted(df_all_loc["Severity"].dropna().unique())
        if severity_vals:
            selected_severity = st.sidebar.multiselect(
                "Filter by Violation Severity", options=severity_vals, default=severity_vals
            )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Data Export")
    export_csv = st.sidebar.checkbox("Enable CSV Export (Raw Data)")
    export_json = st.sidebar.checkbox("Enable JSON Export (Raw Data)")
    
    return {
        "types": selected_types,
        "dates": selected_date_range,
        "severity": selected_severity,
        "export_csv": export_csv,
        "export_json": export_json
    }

# Plotting Functions

def plot_hourly_trend(df_hour):
    """Displays an Altair bar+line chart for hourly trends."""
    if df_hour is None or df_hour.empty:
        st.info("Hourly trend data not available or filtered out.")
        return

    if not {"hour", "total_violations"}.issubset(df_hour.columns):
        st.warning("Hourly data missing 'hour' or 'total_violations' columns.")
        return
        
    st.subheader("Hourly Breakdown") # Use a larger header
    chart_df = df_hour.sort_values("hour")
    base = alt.Chart(chart_df).encode(x=alt.X("hour:Q", title="Hour of Day (24-Hour Clock)"))
    bars = base.mark_bar(opacity=0.8, color="#FF4B4B").encode(
        y=alt.Y("total_violations:Q", title="Total Violations"),
        tooltip=["hour", "total_violations"]
    )
    line = base.mark_line(color="black", strokeWidth=3).encode(y="total_violations:Q")
    
    st.altair_chart(
        # Increased height for bigger visualization
        (bars + line).interactive().properties(height=500, title="Violations by Time of Day"), 
        use_container_width=True
    )

def plot_type_distribution(df_type):
    """Displays a bar chart and table for violation types."""
    if df_type is None or df_type.empty:
        st.info("Violation type data not available or filtered out.")
        return
        
    st.subheader("Violation Type Frequency") # Use a larger header
    df_type_sorted = df_type.sort_values("total_violations", ascending=False)
    
    # Bar chart
    chart = alt.Chart(df_type_sorted.head(15)).mark_bar().encode( # Show top 15 on chart
        x=alt.X("total_violations:Q", title="Total Violations Count"),
        y=alt.Y("Violation_Type:N", sort='-x', title="Violation Type"),
        color=alt.value("#009688"), # Teal color for contrast
        tooltip=["Violation_Type", "total_violations"]
    ).properties(title="Top Violation Types Distribution", height=500) # Increased height
    
    st.altair_chart(chart.interactive(), use_container_width=True)
    
    # Table (all types)
    st.markdown("##### Detailed Breakdown (All Types)")
    st.dataframe(
        df_type_sorted.reset_index(drop=True).style.bar(subset=["total_violations"], color='#d65f5f'),
        use_container_width=True
    )

def plot_time_type_heatmap(df_type_time):
    """Displays a heatmap of violations by type and time window."""
    if df_type_time is None:
        st.info("Type × Time heatmap data not found.")
        return
        
    if not {"Violation_Type", "hour_window", "total_violations"}.issubset(df_type_time.columns):
        st.warning("Heatmap data missing required columns.")
        return
    
    st.subheader("Advanced Analysis: Violation Frequency Heatmap") # Larger header
    heat = alt.Chart(df_type_time).mark_rect().encode(
        x=alt.X("hour_window:O", title="Hour Window"),
        y=alt.Y("Violation_Type:N", title="Violation Type"),
        color=alt.Color("total_violations:Q", scale=alt.Scale(scheme='magma'), title="Violation Count"),
        tooltip=["Violation_Type", "hour_window", "total_violations"]
    ).properties(title="Violation Frequency by Type and Time Window", height=600) # Increased height
    
    st.altair_chart(heat.interactive(), use_container_width=True)

def plot_location_map(df_all_loc_filtered):
    """Displays top locations in a table and on a map (PyDeck or st.map)."""
    # Use the filtered ALL_LOC dataframe for the map, as it's more accurate to filters.
    if df_all_loc_filtered is None or df_all_loc_filtered.empty:
        st.info("Location data not available or filtered out.")
        return

    # --- Data Cleaning ---
    df_all_loc_filtered["Latitude"] = pd.to_numeric(df_all_loc_filtered["Latitude"], errors='coerce')
    df_all_loc_filtered["Longitude"] = pd.to_numeric(df_all_loc_filtered["Longitude"], errors='coerce')

    # Drop rows where Lat/Lon is NaN or exactly 0.0 (common placeholder for missing data)
    map_data = df_all_loc_filtered.dropna(subset=["Latitude", "Longitude"])
    map_data = map_data[(map_data["Latitude"] != 0) & (map_data["Longitude"] != 0)]
    
    # Re-aggregate to get hotspot intensity for the map view
    map_data_agg = map_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='total_violations')


    if map_data_agg.empty:
        st.warning("No valid coordinates found for mapping (Lat/Lon are 0 or null after filtering).")
        return

    # Show table of top 10 locations by count
    st.subheader("Violation Hotspots Table") # Larger header
    st.dataframe(
        map_data_agg.sort_values("total_violations", ascending=False).head(10).reset_index(drop=True)
    )

    # Map
    st.subheader("Spatial Distribution Map")
    if USE_PYDECK:
        # Calculate the mean to center the map
        mid_lat = map_data_agg["Latitude"].mean()
        mid_lon = map_data_agg["Longitude"].mean()
        
        # Use a HeatmapLayer for better density visualization
        layer = pdk.Layer(
            "HeatmapLayer",
            data=map_data_agg,
            opacity=0.8,
            get_position=["Longitude", "Latitude"],
            get_weight="total_violations",
        )
        view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=10, pitch=40)
        r = pdk.Deck(
            layers=[layer], 
            initial_view_state=view_state, 
            map_style="mapbox://styles/mapbox/light-v9"
        )
        # Use the full width for the map
        st.pydeck_chart(r)
    else:
        st.info("PyDeck not installed. Showing basic map (limited to 1000 points).")
        st.map(map_data_agg.head(1000).rename(columns={"Latitude": "lat", "Longitude": "lon"}))
        
def display_kpis(df_all_loc_filtered, df_type_filtered):
    """Displays key performance indicators at the top."""
    
    total_violations = 0
    unique_types = 0
    most_common_type = "N/A"
    
    if df_all_loc_filtered is not None and not df_all_loc_filtered.empty:
        total_violations = len(df_all_loc_filtered)
        
    if df_type_filtered is not None and not df_type_filtered.empty:
        unique_types = len(df_type_filtered)
        most_common_type = df_type_filtered.sort_values("total_violations", ascending=False)["Violation_Type"].iloc[0]
        
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Text is larger by using st.metric
    col1.metric("Total Violations (Filtered)", f"{total_violations:,}", "Data reflects current filters")
    col2.metric("Unique Violation Types", f"{unique_types}", "Count of types after filtering")
    col3.metric("Most Common Violation", most_common_type, "Top type by count")
    
    if df_all_loc_filtered is not None and "Severity" in df_all_loc_filtered.columns:
        # Assuming Severity is numeric 1-5 from prior context, high severity is usually 4 or 5.
        try:
             high_severity_count = df_all_loc_filtered[df_all_loc_filtered["Severity"].astype(int) >= 4].shape[0]
             col4.metric("High Severity Incidents (Sev 4+)", f"{high_severity_count:,}", "Based on filtered data")
        except:
             # Fallback if Severity is not easily comparable
             col4.metric("High Severity Incidents", "N/A", "Severity data type error")
    else:
        col4.metric("High Severity Incidents", "N/A", "Severity data not available")

# Main Application

def main():
    st.title("🚨 Traffic Violation Analysis Dashboard")
    st.markdown("A visual analytics interface powered by **pre-aggregated PySpark outputs**.")

    # 1. Load Data
    data = load_all_data()
    df_hour = data["hour"]
    df_day = data["day"]
    df_type = data["type"]
    df_toploc = data["all_loc"] 
    df_all_loc = data["all_loc"]
    df_type_time = data["type_time"]

    # 2. Setup Sidebar and Get Filters
    filters = setup_sidebar(df_type, df_all_loc)
    
    # 3. Apply Filters
    # --- FILTERING LOGIC ---
    df_type_filtered = df_type
    if df_type is not None and filters["types"]:
        df_type_filtered = df_type[df_type["Violation_Type"].isin(filters["types"])]
    
    df_hour_filtered = df_hour
    if df_hour is not None and "Violation_Type" in df_hour.columns and filters["types"]:
        df_hour_filtered = df_hour[df_hour["Violation_Type"].isin(filters["types"])]
        if "Violation_Type" in df_hour.columns:
             # Re-aggregate after filtering by type
             df_hour_filtered = df_hour_filtered.groupby("hour")["total_violations"].sum().reset_index()
    elif df_hour is not None and df_hour.columns.isin(["hour", "total_violations"]).all():
         pass 
         
    df_toploc_filtered = df_toploc 
    
    df_all_loc_filtered = df_all_loc.copy()
    if df_all_loc_filtered is not None:
        
        if filters["types"] and "Violation_Type" in df_all_loc_filtered.columns:
             df_all_loc_filtered = df_all_loc_filtered[df_all_loc_filtered["Violation_Type"].isin(filters["types"])]
        
        if filters["severity"] and "Severity" in df_all_loc_filtered.columns:
             # Ensure proper type conversion for comparison
             df_all_loc_filtered["Severity"] = pd.to_numeric(df_all_loc_filtered["Severity"], errors='coerce')
             df_all_loc_filtered = df_all_loc_filtered[df_all_loc_filtered["Severity"].isin(filters["severity"])]
        
        if filters["dates"] and len(filters["dates"]) == 2 and "Timestamp" in df_all_loc_filtered.columns:
             start_date = pd.to_datetime(filters["dates"][0])
             end_date = pd.to_datetime(filters["dates"][1])
             df_all_loc_filtered = df_all_loc_filtered[
                 (df_all_loc_filtered["Timestamp"] >= start_date) &
                 (df_all_loc_filtered["Timestamp"] <= end_date + pd.Timedelta(days=1, seconds=-1)) # Include the full end day
             ]
             
    # --- END FILTERING LOGIC ---
    
    # 4. Display KPIs
    st.header("Key Performance Indicators (KPIs)")
    display_kpis(df_all_loc_filtered, df_type_filtered)
    st.markdown("---")


    # 5. Display Main UI (using tabs for neatness)
    tab1, tab2, tab3 = st.tabs(["🕒 Time & Type Analysis", "🗺️ Location Analysis", "🔍 Explore Raw Data"])

    with tab1:
        st.subheader("Time and Violation Type Analysis")
        st.caption("Charts are stacked vertically to maximize clarity and detail across the full page width.")
        
        # --- STACKED LAYOUT ---
        plot_hourly_trend(df_hour_filtered)
        st.markdown("---") # Visual separation
        plot_type_distribution(df_type_filtered)

        st.markdown("---")
        plot_time_type_heatmap(df_type_time) 
        # --- END STACKED LAYOUT ---

    with tab2:
        st.subheader("Violation Hotspots and Spatial Distribution")
        st.caption("The map visualizes the aggregated filtered violation data for hotspot identification.")
        plot_location_map(df_all_loc_filtered)
        
    with tab3:
        st.subheader("Raw Filtered Locations Data Sample")
        st.markdown("Displaying the first 200 rows of the data after applying the current sidebar filters.")
        
        if df_all_loc_filtered is not None:
            st.dataframe(df_all_loc_filtered.head(200), use_container_width=True)
            
            st.markdown("##### 📥 Data Download (Full Filtered Dataset)")
            col_csv, col_json = st.columns(2)
            
            if filters["export_csv"]:
                csv = convert_df_to_bytes(df_all_loc_filtered, "csv")
                col_csv.download_button("Download Full Locations Data (CSV)", data=csv, file_name="all_locations_filtered.csv", mime="text/csv")
            else:
                col_csv.info("Check the 'Enable CSV Export' box in the sidebar to download.")
                
            if filters["export_json"]:
                js = convert_df_to_bytes(df_all_loc_filtered, "json")
                col_json.download_button("Download Full Locations Data (JSON)", data=js, file_name="all_locations_filtered.json", mime="application/json")
            else:
                col_json.info("Check the 'Enable JSON Export' box in the sidebar to download.")
                
        else:
            st.info("Full locations dataset not available.")

    # 6. Export Aggregated Summaries
    st.markdown("---")
    st.subheader("Download Aggregated Summaries (Filtered)")
    st.caption("Download the summarized data used for the charts above.")
    
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        if df_type_filtered is not None:
            st.download_button(
                "⬇️ Type Summary (CSV)", 
                data=convert_df_to_bytes(df_type_filtered, "csv"), 
                file_name="violation_type_summary.csv", 
                mime="text/csv"
            )
    with col_dl2:
        if df_hour_filtered is not None:
            st.download_button(
                "⬇️ Hourly Summary (CSV)", 
                data=convert_df_to_bytes(df_hour_filtered, "csv"), 
                file_name="hourly_summary.csv", 
                mime="text/csv"
            )
    with col_dl3:
        if df_toploc_filtered is not None:
            st.download_button(
                "⬇️ Top Locations Data (CSV)", 
                data=convert_df_to_bytes(df_toploc_filtered, "csv"), 
                file_name="top_locations_data.csv", 
                mime="text/csv"
            )

    # 7. Footer
    st.markdown("---")
    st.caption("Dashboard powered by Streamlit. Data sourced from PySpark pipeline outputs (files in the `output/` folder).")


if __name__ == "__main__":
    main()