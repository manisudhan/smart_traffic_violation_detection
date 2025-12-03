"""
Streamlit dashboard for Traffic Violation Analysis
Reads precomputed parquet outputs created by a PySpark pipeline.
This version includes charts for Day of Week, Monthly, and Day Type comparison.
Includes mock data generation for robust local testing when PySpark outputs are unavailable.
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

# Custom function to display text slightly larger (simulating h2 size for readability)
def display_custom_subheader(text):
    """Uses Markdown to increase text size, similar to a larger H3 or small H2."""
    st.markdown(f"### <p style='font-size: 24px;'>{text}</p>", unsafe_allow_html=True)

# --- MOCK DATA GENERATION ---
def generate_mock_data(key):
    """Generates a small, illustrative DataFrame for testing when Parquet loading fails."""
    
    if key == "hour":
        return pd.DataFrame({
            "hour": range(0, 24),
            "total_violations": np.random.randint(500, 2000, 24),
        })
    elif key == "daily_of_week":
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        return pd.DataFrame({
            "day_of_week": day_order,
            "total_violations": [15000, 16000, 14500, 17000, 20000, 12000, 10000],
        })
    elif key == "monthly":
        return pd.DataFrame({
            "month": range(1, 13),
            "total_violations": np.random.randint(40000, 80000, 12),
        })
    elif key == "daytype":
        return pd.DataFrame({
            "day_type": ['Weekday', 'Weekend'],
            "total_violations": [100000, 45000],
        })
    elif key == "type":
        return pd.DataFrame({
            "Violation_Type": [f"Speeding {i}" for i in range(1, 6)],
            "total_violations": [25000, 18000, 12000, 9000, 5000],
        })
    elif key == "type_time":
        types = [f"Type {i}" for i in range(1, 4)]
        hours = [0, 3, 6, 9, 12, 15, 18, 21]
        data = []
        for t in types:
            for h in hours:
                data.append({
                    "Violation_Type": t,
                    "hour_window": h,
                    "total_violations": np.random.randint(1000, 5000)
                })
        return pd.DataFrame(data)
    elif key == "all_loc":
        # Mock data mimicking raw location data structure
        return pd.DataFrame({
            "Timestamp": pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:30:00', '2024-01-02 14:00:00', '2024-01-03 08:00:00']),
            "Violation_Type": ["Speeding 1", "Illegal Turn", "Speeding 1", "DUI"],
            "Severity": [2, 3, 4, 5],
            "Latitude": [39.0, 39.05, 38.95, 39.1],
            "Longitude": [-77.0, -77.05, -77.1, -77.15],
            "Location": ["Main St", "Side Ave", "Main St", "Highway 40"]
        })
    else:
        return pd.DataFrame()
# --- END MOCK DATA GENERATION ---

# Utility Functions
@st.cache_data(ttl=3600)
def load_parquet(path, key):
    """
    Load parquet into pandas safely. If file/directory is not found, loads mock data.
    """
    is_missing = not os.path.exists(path)
    df = None

    if not is_missing:
        try:
            if os.path.isdir(path):
                files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
                if not files:
                    st.warning(f"Directory {path} contains no .parquet files. Using mock data.")
                    is_missing = True
                else:
                    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            else:
                df = pd.read_parquet(path)
        except Exception as e:
            st.error(f"Error reading {path}: {e}. Using mock data.")
            is_missing = True
    
    if is_missing or df is None:
        st.info(f"Could not load data for '{key}' from '{path}'. Displaying **mock data**.")
        return generate_mock_data(key)

    return df


def convert_df_to_bytes(df, fmt="csv"):
    """Utility to convert DataFrame to bytes for downloading."""
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    else:
        return df.to_json(orient="records", date_format="iso").encode("utf-8")

@st.cache_data(ttl=3600)
def load_all_data():
    """Loads all required and optional dataframes, using mock data as fallback."""
    with st.spinner("⏳ Loading PySpark data outputs (or generating mock data)..."):
        data = {
            # Time-based outputs
            "hour": load_parquet("output/time_based/hourly", "hour"),
            "daily_of_week": load_parquet("output/time_based/daily_of_week", "daily_of_week"), 
            "monthly": load_parquet("output/time_based/monthly", "monthly"), 
            "yearly": load_parquet("output/time_based/yearly", "yearly"), 
            
            # Offense/Type outputs
            "type": load_parquet("output/offense_type/type_summary", "type"),
            "type_time": load_parquet("output/advanced/type_time", "type_time"),
            
            # Location outputs
            "top_loc": load_parquet("output/location_based/top_locations", "top_loc"), # Not strictly used, but kept for consistency
            "all_loc": load_parquet("output/location_based/all_locations", "all_loc"),
            
            # Advanced outputs
            "daytype": load_parquet("output/advanced/daytype", "daytype"), # Weekday vs Weekend
        }
    return data

def setup_sidebar(df_type, df_all_loc):
    """Creates the sidebar filters and returns selected values."""
    st.sidebar.header("🗄️ Filter Options")
    
    # 1. Violation Type filter
    if df_type is not None and "Violation_Type" in df_type.columns:
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
        # Create a copy to prevent Streamlit error on cached dataframe modification
        df_temp = df_all_loc.copy() 
        try:
             df_temp["Timestamp"] = pd.to_datetime(df_temp["Timestamp"], errors="coerce")
             min_date = df_temp["Timestamp"].min()
             max_date = df_temp["Timestamp"].max()
        except:
             min_date = pd.NaT
             max_date = pd.NaT
        
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
        severity_vals = sorted(df_all_loc["Severity"].dropna().astype(str).unique())
        if severity_vals:
            try:
                # Try to convert to int for better display/sorting
                severity_vals_int = [int(x) for x in severity_vals if x.isdigit()]
                if severity_vals_int:
                     severity_vals = sorted(severity_vals_int)
            except:
                pass # Keep as strings if conversion fails
                
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
        
    st.subheader("Hourly Breakdown")
    chart_df = df_hour.sort_values("hour")
    base = alt.Chart(chart_df).encode(x=alt.X("hour:Q", title="Hour of Day (24-Hour Clock)",axis=alt.Axis(titleFontSize=25, labelFontSize=15,labelColor="black")))
    
    bars = base.mark_bar(opacity=0.8, color="#1f77b4").encode( # Deep Blue
        y=alt.Y("total_violations:Q", title="Total Violations",axis=alt.Axis(titleFontSize=25, labelFontSize=15,labelColor="black")),
        tooltip=["hour", "total_violations"]
    )
    line = base.mark_line(color="black", strokeWidth=3).encode(y="total_violations:Q")
    
    st.altair_chart(
        (bars + line).interactive().properties(height=400, title="Violations by Time of Day"), 
        use_container_width=True
    )
    
def plot_violations_by_day_of_week(df_daily):
    """Displays a bar chart for violations by day of the week."""
    if df_daily is None or df_daily.empty:
        st.info("Daily-of-week data not available.")
        return
    
    st.subheader("Violations by Day of Week 🗓️")
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    chart_df = df_daily.copy()
    if 'day_of_week' in chart_df.columns:
        chart_df = chart_df[chart_df['day_of_week'].isin(day_order)]
        chart_df['day_of_week'] = pd.Categorical(chart_df['day_of_week'], categories=day_order, ordered=True)
        chart_df = chart_df.sort_values('day_of_week')
        
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("day_of_week:N", sort=day_order, title="Day of Week", axis=alt.Axis(titleFontSize=25, labelFontSize=15, labelColor="black")),
            y=alt.Y("total_violations:Q", title="Total Violations", axis=alt.Axis(titleFontSize=25, labelFontSize=15, labelColor="black")),
            color=alt.condition(
                alt.FieldOneOf("day_of_week", ["Sat", "Sun"], type="nominal"),
                alt.value("firebrick"),  # Weekend color
                alt.value("olivedrab")   # Weekday color
            ),
            tooltip=["day_of_week", "total_violations"]
        ).properties(title="Violation Count by Day of the Week", height=400)
        
        st.altair_chart(chart.interactive(), use_container_width=True)
    else:
        st.warning("Daily-of-week data is missing the 'day_of_week' column.")

def plot_monthly_trend(df_monthly):
    """Displays a line chart for violations per month."""
    if df_monthly is None or df_monthly.empty:
        st.info("Monthly trend data not available.")
        return
        
    st.subheader("Seasonal Trend: Violations per Month")
    month_map = {i: pd.to_datetime(i, format='%m').strftime('%b') for i in range(1, 13)}
    
    chart_df = df_monthly.copy()
    if 'month' in chart_df.columns:
        chart_df['month_name'] = chart_df['month'].apply(lambda x: month_map.get(x, str(x)))
        
        base = alt.Chart(chart_df).encode(
            x=alt.X("month:O", title="Month", axis=alt.Axis(labelAngle=0, titleFontSize=25, labelFontSize=15, labelColor="black")),
            y=alt.Y("total_violations:Q", title="Total Violations", axis=alt.Axis(titleFontSize=25, labelFontSize=15, labelColor="black")),
            tooltip=["month_name", "total_violations"]
        )
        
        line = base.mark_line(color="#7c1158", strokeWidth=3).encode(
            order=alt.Order('month:Q')
        )
        points = base.mark_point(filled=True, size=60, color="#7c1158")
        
        st.altair_chart(
            (line + points).interactive().properties(height=400, title="Violations by Month"), 
            use_container_width=True
        )
    else:
        st.warning("Monthly data is missing the 'month' column.")

def plot_day_type_comparison(df_daytype):
    """Displays a simple bar chart comparing Weekday vs Weekend violations."""
    if df_daytype is None or df_daytype.empty:
        st.info("Weekday/Weekend comparison data not available.")
        return
    
    st.subheader("Weekday vs. Weekend Comparison ⚖️")
    
    chart = alt.Chart(df_daytype).mark_bar().encode(
        x=alt.X("day_type:N", title="Day Type"),
        y=alt.Y("total_violations:Q", title="Total Violations"),
        color=alt.Color("day_type:N", scale=alt.Scale(domain=['Weekday', 'Weekend'], range=['#32CD32', '#FF4500'])),
        tooltip=["day_type", "total_violations"]
    ).properties(title="Violations Count: Weekday vs Weekend", height=400)
    
    st.altair_chart(chart, use_container_width=True)

def plot_type_distribution(df_type):
    """Displays a bar chart and table for violation types."""
    if df_type is None or df_type.empty:
        st.info("Violation type data not available or filtered out.")
        return
        
    st.subheader("Violation Type Frequency")
    df_type_sorted = df_type.sort_values("total_violations", ascending=False)
    
    chart = alt.Chart(df_type_sorted.head(15)).mark_bar().encode( 
        x=alt.X("total_violations:Q", title="Total Violations Count",axis=alt.Axis(titleFontSize=25, labelFontSize=15,labelColor="black")),
        y=alt.Y("Violation_Type:N", sort='-x', title="Violation Type",axis=alt.Axis(titleFontSize=25, labelFontSize=15,labelColor="black")),
        color=alt.value("#2ca02c"), 
        tooltip=["Violation_Type", "total_violations"]
    ).properties(title="Top 15 Violation Types Distribution", height=500)
    
    st.altair_chart(chart.interactive(), use_container_width=True)
    
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
    
    st.subheader("Advanced Analysis: Violation Frequency Heatmap")
    heat = alt.Chart(df_type_time).mark_rect().encode(
        x=alt.X("hour_window:O", title="Hour Window"),
        y=alt.Y("Violation_Type:N", title="Violation Type"),
        color=alt.Color("total_violations:Q", scale=alt.Scale(scheme='magma'), title="Violation Count"),
        tooltip=["Violation_Type", "hour_window", "total_violations"]
    ).properties(title="Violation Frequency by Type and Time Window", height=600)
    
    st.altair_chart(heat.interactive(), use_container_width=True)

def plot_location_map(df_all_loc_filtered):
    """Plots a PyDeck heatmap of violation locations."""
    if df_all_loc_filtered is None or df_all_loc_filtered.empty:
        st.info("Location data not available or filtered out.")
        return

    import pydeck as pdk

    MAPBOX_API_KEY = st.secrets.get("MAPBOX_API_KEY", None)

    if MAPBOX_API_KEY is None and not USE_PYDECK:
        st.error("PyDeck failed to load or Mapbox API key is missing in secrets.")
        return
    
    if USE_PYDECK:
        # Check if mapbox key is available (it's often required for PyDeck on cloud platforms)
        if MAPBOX_API_KEY:
            os.environ["MAPBOX_API_KEY"] = MAPBOX_API_KEY
        
        # --- Data Cleaning (Copy for safe local operation) ---
        map_data_source = df_all_loc_filtered.copy()
        map_data_source["Latitude"] = pd.to_numeric(map_data_source["Latitude"], errors='coerce')
        map_data_source["Longitude"] = pd.to_numeric(map_data_source["Longitude"], errors='coerce')

        map_data = map_data_source.dropna(subset=["Latitude", "Longitude"])
        map_data = map_data[(map_data["Latitude"] != 0) & (map_data["Longitude"] != 0)]

        map_data_agg = map_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='total_violations')

        if map_data_agg.empty:
            st.warning("No valid coordinates available after filtering for map display.")
            return

        st.subheader("Violation Hotspots Table (Top 10 Coords)")
        st.dataframe(
            map_data_agg.sort_values("total_violations", ascending=False).head(10).style.format({"Latitude": "{:.4f}", "Longitude": "{:.4f}"}),
            use_container_width=True
        )

        st.subheader("Spatial Distribution Map")

        mid_lat = map_data_agg["Latitude"].mean()
        mid_lon = map_data_agg["Longitude"].mean()

        layer = pdk.Layer(
            "HeatmapLayer",
            data=map_data_agg,
            get_position=["Longitude", "Latitude"],
            get_weight="total_violations",
            opacity=0.9,
        )

        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=10,
            pitch=40
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_provider="mapbox",
            map_style="mapbox://styles/mapbox/dark-v9",
        )

        st.pydeck_chart(r)
    else:
        st.error("PyDeck is required for the map visualization but failed to import. Check installation or environment.")


def plot_top_locations_bar(df_all_loc_filtered, top_n=15):
    """
    Displays a bar chart for the top N violation locations (e.g., street names).
    Aggregates the filtered raw location data.
    """
    if df_all_loc_filtered is None or df_all_loc_filtered.empty:
        st.info("Location data not available or filtered out.")
        return
    
    location_col = None
    if 'Location' in df_all_loc_filtered.columns:
        location_col = 'Location'
    elif 'Street' in df_all_loc_filtered.columns:
        location_col = 'Street'
    elif 'Location_Name' in df_all_loc_filtered.columns:
        location_col = 'Location_Name'
    
    if location_col is None:
        st.error("Cannot find a suitable location column ('Location', 'Street', or 'Location_Name') for the bar chart.")
        return

    df_toploc_agg = df_all_loc_filtered.groupby(location_col).size().reset_index(name='total_violations')
    df_toploc_agg_sorted = df_toploc_agg.sort_values("total_violations", ascending=False).head(top_n)

    if df_toploc_agg_sorted.empty:
        st.info(f"No valid locations found in the top {top_n} after filtering.")
        return

    st.subheader(f"Top {top_n} Violation Locations (Non-Map)")
    
    chart = alt.Chart(df_toploc_agg_sorted).mark_bar().encode(
        x=alt.X("total_violations:Q", title="Total Violations Count", axis=alt.Axis(titleFontSize=25, labelFontSize=15, labelColor="black")),
        y=alt.Y(alt.Field(location_col, type="nominal"), sort='-x', title="Location/Street Name", axis=alt.Axis(titleFontSize=25, labelFontSize=15, labelColor="black")),
        color=alt.value("#d62728"), 
        tooltip=[alt.Field(location_col, title="Location"), "total_violations"]
    ).properties(title=f"Top {top_n} Locations by Violation Count", height=500)
    
    st.altair_chart(chart.interactive(), use_container_width=True)


def display_kpis(df_all_loc_filtered, df_type_filtered):
    """Displays key performance indicators at the top."""
    total_violations = 0
    unique_types = 0
    most_common_type = "N/A"
    
    if df_all_loc_filtered is not None and not df_all_loc_filtered.empty:
        total_violations = len(df_all_loc_filtered)
        
    if df_type_filtered is not None and not df_type_filtered.empty:
        unique_types = len(df_type_filtered)
        if "Violation_Type" in df_type_filtered.columns:
            most_common_type = df_type_filtered.sort_values("total_violations", ascending=False)["Violation_Type"].iloc[0]
        
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Violations (Filtered)", f"{total_violations:,}", "")
    col2.metric("Unique Violation Types", f"{unique_types}", "")
    col3.metric("Most Common Violation", most_common_type, "")
    
    if df_all_loc_filtered is not None and "Severity" in df_all_loc_filtered.columns:
        try:
             # Create a temporary copy to perform numeric conversion without warning
             temp_df = df_all_loc_filtered.copy()
             temp_df["Severity"] = pd.to_numeric(temp_df["Severity"], errors='coerce')
             high_severity_count = temp_df[temp_df["Severity"] >= 4].shape[0]
             col4.metric("High Severity Incidents (Sev 4+)", f"{high_severity_count:,}", "")
        except:
             col4.metric("High Severity Incidents", "N/A (Error)", "")
    else:
        col4.metric("High Severity Incidents", "N/A", "")

# Main Application

def main():

    st.title("🚨 Traffic Violation Analysis Dashboard")
    st.markdown("A visual analytics interface powered by **pre-aggregated PySpark outputs**.")

    # 1. Load Data
    data = load_all_data()
    df_hour = data["hour"]
    df_daily_of_week = data["daily_of_week"] 
    df_monthly = data["monthly"]             
    df_yearly = data["yearly"]               
    df_daytype = data["daytype"]             
    df_type = data["type"]
    df_all_loc = data["all_loc"]
    df_type_time = data["type_time"]

    # 2. Setup Sidebar and Get Filters
    df_type_for_sidebar = df_type.copy() if df_type is not None else None
    df_all_loc_for_sidebar = df_all_loc.copy() if df_all_loc is not None else None
    filters = setup_sidebar(df_type_for_sidebar, df_all_loc_for_sidebar)
    
    # 3. Apply Filters
    
    df_type_filtered = df_type
    if df_type is not None and filters["types"]:
        df_type_filtered = df_type[df_type["Violation_Type"].isin(filters["types"])]
    
    # Filter df_all_loc (The source for map/hotspot and KPI counts)
    df_all_loc_filtered = df_all_loc.copy()
    if df_all_loc_filtered is not None:
        
        if filters["types"] and "Violation_Type" in df_all_loc_filtered.columns:
             df_all_loc_filtered = df_all_loc_filtered[df_all_loc_filtered["Violation_Type"].isin(filters["types"])]
        
        if filters["severity"] and "Severity" in df_all_loc_filtered.columns:
             # Convert filtered severity list items to string for comparison
             str_severity_list = [str(s) for s in filters["severity"]]
             df_all_loc_filtered = df_all_loc_filtered[df_all_loc_filtered["Severity"].astype(str).isin(str_severity_list)]
        
        if filters["dates"] and len(filters["dates"]) == 2 and "Timestamp" in df_all_loc_filtered.columns:
             start_date = pd.to_datetime(filters["dates"][0])
             end_date = pd.to_datetime(filters["dates"][1])
             
             # Ensure Timestamp is datetime before filtering
             df_all_loc_filtered["Timestamp"] = pd.to_datetime(df_all_loc_filtered["Timestamp"], errors='coerce')

             df_all_loc_filtered = df_all_loc_filtered[
                 (df_all_loc_filtered["Timestamp"] >= start_date) &
                 (df_all_loc_filtered["Timestamp"] <= end_date + pd.Timedelta(days=1, seconds=-1))
             ]
             
    # --- END FILTERING LOGIC ---
    
    # 4. Display KPIs
    st.header("Key Performance Indicators (KPIs)")
    # Pass copies to KPI function to prevent modifying the filtered dataframe 
    # (especially due to mock data issues)
    display_kpis(df_all_loc_filtered.copy(), df_type_filtered.copy())
    st.markdown("---")


    # 5. Display Main UI (using tabs for neatness)
    tab1, tab2, tab3 = st.tabs(["🕒 Time & Type Analysis", "🗺️ Location Analysis", "🔍 Explore Raw Data"])

    with tab1:
        display_custom_subheader("Time and Violation Type Analysis")
        st.caption("Charts visualize violation patterns across different time dimensions (note: aggregated time charts are not filtered by Type/Severity).")
        
        # Row 1: Hourly and Daily-of-Week
        col1, col2 = st.columns(2)
        with col1:
            plot_hourly_trend(df_hour)
        with col2:
            plot_violations_by_day_of_week(df_daily_of_week) 
        
        st.markdown("---")
        
        # Row 2: Monthly and Day Type Comparison
        col3, col4 = st.columns(2)
        with col3:
            plot_monthly_trend(df_monthly)
        with col4:
            plot_day_type_comparison(df_daytype)
        
        st.markdown("---") 
        st.markdown("## Violation Type and Time Correlation")
        plot_type_distribution(df_type_filtered)

        st.markdown("---")
        plot_time_type_heatmap(df_type_time) 

    with tab2:
        display_custom_subheader("Violation Hotspots and Spatial Distribution")
        st.caption("The map visualizes the aggregated filtered violation data for hotspot identification.")
        
        plot_top_locations_bar(df_all_loc_filtered)
        
        st.markdown("---")
        plot_location_map(df_all_loc_filtered)
        
    with tab3:
        display_custom_subheader("Raw Filtered Locations Data Sample")
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
    st.subheader("Download Aggregated Summaries (Unfiltered)")
    st.caption("Download the summarized data used for the charts above. Note: These are often derived from the full, unfiltered dataset for overall trends.")
    
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
    
    with col_dl1:
        if df_type is not None:
            st.download_button(
                "⬇️ Type Summary (CSV)", 
                data=convert_df_to_bytes(df_type, "csv"), 
                file_name="violation_type_summary.csv", 
                mime="text/csv"
            )
    with col_dl2:
        if df_hour is not None:
            st.download_button(
                "⬇️ Hourly Summary (CSV)", 
                data=convert_df_to_bytes(df_hour, "csv"), 
                file_name="hourly_summary.csv", 
                mime="text/csv"
            )
    with col_dl3:
        if df_daily_of_week is not None:
            st.download_button(
                "⬇️ Daily-of-Week Summary (CSV)", 
                data=convert_df_to_bytes(df_daily_of_week, "csv"), 
                file_name="daily_of_week_summary.csv", 
                mime="text/csv"
            )
    with col_dl4:
        if df_monthly is not None:
            st.download_button(
                "⬇️ Monthly Summary (CSV)", 
                data=convert_df_to_bytes(df_monthly, "csv"), 
                file_name="monthly_summary.csv", 
                mime="text/csv"
            )


    # 7. Footer
    st.markdown("---")
    st.caption("Dashboard powered by Streamlit. Data sourced from PySpark pipeline outputs (files in the `output/` folder).")


if __name__ == "__main__":
    main()