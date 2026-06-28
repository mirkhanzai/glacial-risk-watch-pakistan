import json
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Northern Pakistan Glacial Risk Watch",
    layout="wide"
)

st.title("Northern Pakistan Glacial Flood, Heat and Landslide Watch")

st.warning(
    "Automated decision-support dashboard only. Verify all alerts with NDMA, PMD, "
    "GBDMA, PDMA KP, district administration, Rescue 1122, and field observers."
)

latest_path = Path("data/latest_result.json")
history_path = Path("data/history.csv")
valleys_path = Path("data/latest_valleys.csv")

if not latest_path.exists():
    st.error("No monitoring result found yet. Run main.py or GitHub Action first.")
    st.stop()

with open(latest_path, "r") as f:
    latest = json.load(f)

combined = latest["combined_risk"]

col1, col2, col3 = st.columns(3)

col1.metric("Overall Risk Level", combined["risk_level"])
col2.metric("Overall Risk Score", combined["score"])
col3.metric("Last Update UTC", latest["timestamp"])

st.subheader("Valley Risk Map")

if valleys_path.exists():
    vdf = pd.read_csv(valleys_path)

    center_lat = vdf["lat"].mean()
    center_lon = vdf["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap")

    color_map = {
        "NORMAL": "green",
        "WATCH": "orange",
        "WARNING": "red",
        "CRITICAL": "darkred",
        "UNKNOWN": "gray"
    }

    for _, row in vdf.iterrows():
        color = color_map.get(row["risk_level"], "gray")

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=f"{row['valley']} | {row['risk_level']} | Score: {row['score']}"
        ).add_to(m)

    st_folium(m, width=1200, height=500)

st.subheader("Latest Valley Results")

valley_rows = []

for valley in latest["valleys"]:
    valley_rows.append({
        "Valley": valley["name"],
        "Risk Level": valley["combined_risk"]["risk_level"],
        "Score": valley["combined_risk"]["score"],
        "Latitude": valley["lat"],
        "Longitude": valley["lon"]
    })

st.dataframe(pd.DataFrame(valley_rows), use_container_width=True)

st.subheader("Detailed Agent Results")

for valley in latest["valleys"]:
    with st.expander(f"{valley['name']} — {valley['combined_risk']['risk_level']}"):
        if valley.get("error"):
            st.error(valley["error"])
        else:
            rows = []
            for agent in valley["agents"]:
                rows.append({
                    "Agent": agent["agent"],
                    "Risk": agent["risk"],
                    "Summary": agent["summary"]
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.subheader("Risk Timeline")

if history_path.exists():
    hdf = pd.read_csv(history_path)

    fig = px.line(
        hdf,
        x="timestamp",
        y="score",
        color="risk_level",
        title="Overall Risk Score Timeline"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No history file found yet.")