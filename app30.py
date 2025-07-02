# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:18:38 2025

@author: z00534vd
"""

import streamlit as st
import requests
import pandas as pd
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings (unsafe for production use)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

st.title("BPEC Live Stint Analysis")
st.markdown("Analyze stints for a given kart based on pitstop threshold.")

# --- User Inputs ---
kart_number = st.text_input("Enter Competitor Number (e.g. '15')", value="15")
pit_threshold_sec = st.number_input("Pitstop Threshold (seconds)", min_value=60, max_value=300, value=80)

# --- Fetch JSON from API ---
url = "https://results.alphatiming.co.uk/api/v1/bpec/live/current"

try:
    response = requests.get(url, verify=False, timeout=10)
    response.raise_for_status()
    data = response.json()
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# --- Extract Competitor Data ---
competitors = data.get("Competitors", [])
team = next((c for c in competitors if c.get("CompetitorNumber") == kart_number), None)

if not team:
    st.warning(f"No data found for CompetitorNumber {kart_number}")
    st.stop()

laps = team.get("Laps", [])

if not laps:
    st.warning("No lap data available for this team.")
    st.stop()

# --- Convert to DataFrame ---
df_laps = pd.DataFrame(laps)

relevant_cols = [
    "LapNumber", "LapTime", "Position", "PositionChange",
    "Gap", "Behind", "Split1Time", "Split2Time", "Split3Time"
]
cols_to_use = [col for col in relevant_cols if col in df_laps.columns]
df_laps = df_laps[cols_to_use]

df_laps["LapTime"] = pd.to_numeric(df_laps["LapTime"], errors="coerce")
df_laps["LapNumber"] = pd.to_numeric(df_laps["LapNumber"], errors="coerce")

df_laps = df_laps.sort_values("LapNumber").reset_index(drop=True)

# --- Stint Analysis ---
PIT_STOP_THRESHOLD = pit_threshold_sec * 1000  # convert to milliseconds

df_laps["IsPitStop"] = df_laps["LapTime"] > PIT_STOP_THRESHOLD
df_laps["StintNumber"] = df_laps["IsPitStop"].cumsum()

stint_summary = []

for stint_num, group in df_laps.groupby("StintNumber"):
    laps_in_stint = group[~group["IsPitStop"]]
    if laps_in_stint.empty:
        continue

    start_lap = laps_in_stint["LapNumber"].min()
    end_lap = laps_in_stint["LapNumber"].max()
    stint_len = len(laps_in_stint)
    stint_time_min = laps_in_stint["LapTime"].sum() / 60000

    best_lap_sec = laps_in_stint["LapTime"].min() / 1000
    median_lap_sec = laps_in_stint["LapTime"].median() / 1000

    pit_lap_time_ms = group[group["IsPitStop"]]["LapTime"].iloc[-1] if not group[group["IsPitStop"]].empty else None
    pit_lap_time_sec = round(pit_lap_time_ms / 1000, 3) if pit_lap_time_ms is not None else None

    stint_summary.append({
        "StintNumber": stint_num,
        "StartLap": int(start_lap),
        "EndLap": int(end_lap),
        "LapsInStint": stint_len,
        "StintTime_Min": round(stint_time_min, 2),
        "BestLap_sec": round(best_lap_sec, 3),
        "MedianLap_sec": round(median_lap_sec, 3),
        "PitLapTime_sec": pit_lap_time_sec
    })

df_stints = pd.DataFrame(stint_summary)

# --- Output Table ---
st.subheader(f"Stint Summary for Kart #{kart_number}")
st.dataframe(df_stints)
