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


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="🚨 Traffic Violation Dashboard | Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------
@st.cache_data(ttl=3600)
def load_parquet(path):
    """Load parquet files or directories safely."""
    if not os.path.exists(path):
        return None

    try:
        if os.path.isdir(path):
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".parquet")
            ]
            if not files:
                return None
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        else:
            return pd.read_parquet(path)

    except Exception as e:
        st.error(f"Error reading parquet {path}: {e}")
        return None


@st.cache_data(ttl=3600)
def load_all_data():
    """Loads all required parquet outputs."""
    with st.spinner("⏳ Loading PySpark output files..."):
        return {
            "hour": load_parquet("output/time_based/hourly"),
            "day": load_parquet("output/time_based/daily"),
            "type": load_parquet("output/offense_type/type_summary"),
            "top_loc": load_parquet("output/location_based/top_locations"),
            "all_loc": load_parquet("output/location_based/all_locations"),
            "grid": load_parquet("output/advanced/spatial_grid"),
            "type_time": load_parquet("output/advanced/type_time"),
        }


def convert_df_to_bytes(df, fmt="csv"):
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8")
    return df.to_json(orient="records").encode("utf-8")


def display_custom_subheader(text):
    st.markdown(f"### <p style='font-size: 24px;'>{text}</p>", unsafe_allow_html=True)


# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------
def setup_sidebar(df_type, df_all):
    st.sidebar.header("🗄️ Filter Options")

    # Violation Type Filter
    selected_types = []
    if df_type is not None:
        options = sorted(df_type["Violation_Type"].unique())
        selected_types = st.sidebar.multiselect(
            "Select Violation Types", options, options
        )

    # Date Range Filter
    selected_date_range = None
    if df_all is not None and "Timestamp" in df_all.columns:
        df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")

        min_date = df_all["Timestamp"].min()
        max_date = df_all["Timestamp"].max()

        if pd.notna(min_date) and pd.notna(max_date):
            selected_date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )

    # Severity filter
    selected_severity = None
    if df_all is not None and "Severity" in df_all.columns:
        sev = sorted(df_all["Severity"].dropna().unique())
        selected_severity = st.sidebar.multiselect("Severity Levels", sev, sev)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Export")
    exp_csv = st.sidebar.checkbox("Enable CSV Export")
    exp_json = st.sidebar.checkbox("Enable JSON Export")

    return {
        "types": selected_types,
        "dates": selected_date_range,
        "severity": selected_severity,
        "export_csv": exp_csv,
        "export_json": exp_json,
    }


# -------------------------------------------------------
# ORIGINAL PLOTS + NEW ADDITIONS
# -------------------------------------------------------

def plot_hourly_trend(df):
    if df is None or df.empty:
        st.info("Hourly data missing.")
        return

    st.subheader("Hourly Trend")
    c = alt.Chart(df).mark_line(point=True).encode(
        x="hour:Q",
        y="total_violations:Q",
        tooltip=["hour", "total_violations"]
    ).properties(height=350)

    st.altair_chart(c, use_container_width=True)


def plot_type_distribution(df):
    if df is None or df.empty:
        st.info("Violation type data missing.")
        return

    st.subheader("Violation Type Distribution")

    c = alt.Chart(df).mark_bar().encode(
        x="total_violations:Q",
        y=alt.Y("Violation_Type:N", sort="-x"),
        tooltip=["Violation_Type", "total_violations"]
    )

    st.altair_chart(c, use_container_width=True)


def plot_weekday_trend(df_day):
    if df_day is None or df_day.empty or "day_of_week" not in df_day.columns:
        return

    st.subheader("Violations by Day of Week")

    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    c = alt.Chart(df_day).mark_bar(color="#FF7F50").encode(
        x=alt.X("day_of_week:N", sort=order),
        y="total_violations:Q",
        tooltip=["day_of_week", "total_violations"]
    )

    st.altair_chart(c, use_container_width=True)


def plot_time_type_heatmap(df):
    if df is None or df.empty:
        return

    st.subheader("Violation Type × Time Window Heatmap")

    c = alt.Chart(df).mark_rect().encode(
        x="hour_window:O",
        y="Violation_Type:N",
        color=alt.Color("total_violations:Q", scale=alt.Scale(scheme="magma")),
        tooltip=["Violation_Type", "hour_window", "total_violations"]
    )

    st.altair_chart(c, use_container_width=True)


def plot_hour_severity_heatmap(df):
    if df is None or df.empty:
        return

    if "Timestamp" not in df.columns:
        return

    df = df.copy()
    df["hour"] = df["Timestamp"].dt.hour

    pivot = df.groupby(["hour", "Severity"]).size().reset_index(name="count")

    st.subheader("Hour × Severity Heatmap")

    c = alt.Chart(pivot).mark_rect().encode(
        x="hour:O",
        y="Severity:O",
        color=alt.Color("count:Q", scale=alt.Scale(scheme="inferno")),
        tooltip=["hour", "Severity", "count"]
    )

    st.altair_chart(c, use_container_width=True)


def plot_day_hour_heatmap(df):
    if df is None or df.empty:
        return

    if "Timestamp" not in df.columns:
        return

    df = df.copy()
    df["hour"] = df["Timestamp"].dt.hour
    df["day"] = df["Timestamp"].dt.day_name()

    pivot = df.groupby(["day", "hour"]).size().reset_index(name="count")

    st.subheader("Day × Hour Heatmap")

    c = alt.Chart(pivot).mark_rect().encode(
        x="hour:O",
        y=alt.Y("day:N", sort=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
        color=alt.Color("count:Q", scale=alt.Scale(scheme="magma")),
        tooltip=["day","hour","count"]
    )

    st.altair_chart(c, use_container_width=True)


# -------------------------------------------------------
# KPIs
# -------------------------------------------------------
def display_kpis(df_all, df_type):
    st.header("Key Performance Indicators")

    total = len(df_all)
    unique_types = len(df_type) if df_type is not None else 0
    severe = df_all[df_all["Severity"] >= 4].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Violations", f"{total:,}")
    c2.metric("Unique Violation Types", unique_types)
    c3.metric("High Severity (4+)", f"{severe:,}")

    if "Timestamp" in df_all.columns:
        latest = df_all["Timestamp"].max()
        c4.metric("Last Updated", str(latest.date()))


# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------
def main():

    st.title("🚨 Traffic Violation Analytics Dashboard")
    st.markdown("Powered by PySpark + Streamlit")

    # Load all data
    d = load_all_data()
    df_hour = d["hour"]
    df_day = d["day"]
    df_type = d["type"]
    df_all = d["all_loc"]
    df_type_time = d["type_time"]

    # Filters
    filters = setup_sidebar(df_type, df_all)

    # Apply filtering
    df_all_f = df_all.copy()

    if filters["types"]:
        df_all_f = df_all_f[df_all_f["Violation_Type"].isin(filters["types"])]

    if filters["severity"]:
        df_all_f = df_all_f[df_all_f["Severity"].isin(filters["severity"])]

    if filters["dates"] and len(filters["dates"]) == 2:
        start = pd.to_datetime(filters["dates"][0])
        end = pd.to_datetime(filters["dates"][1])
        df_all_f = df_all_f[
            (df_all_f["Timestamp"] >= start)
            & (df_all_f["Timestamp"] <= end)
        ]

    # KPIs
    display_kpis(df_all_f, df_type)

    st.markdown("---")

    # TABS
    tab1, tab2, tab3 = st.tabs([
        "🕒 Time & Type Analysis",
        "🗺️ Location Analysis",
        "📄 Raw Data"
    ])

    # ----------------------------
    # TAB 1 — TIME & TYPE ANALYSIS
    # ----------------------------
    with tab1:
        display_custom_subheader("Time & Violation Patterns")

        plot_hourly_trend(df_hour)
        st.markdown("---")

        plot_weekday_trend(df_day)
        st.markdown("---")

        plot_type_distribution(df_type)
        st.markdown("---")

        plot_time_type_heatmap(df_type_time)
        st.markdown("---")

        plot_hour_severity_heatmap(df_all_f)
        st.markdown("---")

        plot_day_hour_heatmap(df_all_f)

    # ----------------------------
    # TAB 2 — LOCATION
    # ----------------------------
    with tab2:
        st.subheader("Violation Hotspots (Top 10)")
        df_top = d["top_loc"]
        if df_top is not None:
            st.dataframe(df_top)

        if USE_PYDECK:
            st.subheader("Geo Heatmap of Violations")

            dfm = df_all_f.dropna(subset=["Latitude","Longitude"])

            heatmap = pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=dfm["Latitude"].mean(),
                    longitude=dfm["Longitude"].mean(),
                    zoom=11
                ),
                layers=[
                    pdk.Layer(
                        "HeatmapLayer",
                        data=dfm,
                        get_position=["Longitude","Latitude"],
                        radiusPixels=40
                    )
                ]
            )
            st.pydeck_chart(heatmap)
        else:
            st.info("pydeck not available. Install via pip to enable maps.")

    # ----------------------------
    # TAB 3 — RAW DATA
    # ----------------------------
    with tab3:
        st.subheader("Raw Filtered Data")
        st.dataframe(df_all_f.head(200), use_container_width=True)

        if filters["export_csv"]:
            st.download_button(
                "Download CSV",
                convert_df_to_bytes(df_all_f),
                "filtered_data.csv",
                "text/csv"
            )

        if filters["export_json"]:
            st.download_button(
                "Download JSON",
                convert_df_to_bytes(df_all_f, fmt="json"),
                "filtered_data.json",
                "application/json"
            )


# Run
if __name__ == "__main__":
    main()
