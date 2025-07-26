# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 09:18:38 2025

@author: z00534vd
"""

import streamlit as st
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

def format_seconds(val):
    if pd.isna(val) or val is None:
        return ""
    s = f"{val:.3f}"
    s = s.rstrip('0').rstrip('.') if '.' in s else s
    return s

st.set_page_config(page_title="BPEC Live Analysis", layout="wide")

st.title("BPEC Live Analysis")

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Initialize session state for accumulated laps dataframe
if 'all_laps_df' not in st.session_state:
    st.session_state['all_laps_df'] = pd.DataFrame()

# Fetch live data function
@st.cache_data(ttl=25)  # small cache to avoid hammering source during quick reruns
def fetch_live_data():
    #url = "https://results.alphatiming.co.uk/api/v1/bpec/live/current"
    url = "https://results.alphatiming.co.uk/api/v1/bukc/live/current"

    response = requests.get(url, verify=False)
    response.raise_for_status()
    return response.json()

# Fetch new live data
data = fetch_live_data()

# Flatten laps from all competitors
new_rows = []
for comp in data["Competitors"]:
    kart = comp.get("CompetitorNumber")
    laps = comp.get("Laps", [])
    for lap in laps:
        lap["CompetitorNumber"] = kart
        new_rows.append(lap)

df_new = pd.DataFrame(new_rows)
df_new["LapNumber"] = pd.to_numeric(df_new["LapNumber"], errors='coerce')

# Append only new laps (by kart + lap number)
existing = st.session_state['all_laps_df']
if not existing.empty:
    existing_keys = set(zip(existing["CompetitorNumber"], existing["LapNumber"]))
    new_keys = set(zip(df_new["CompetitorNumber"], df_new["LapNumber"]))

    to_add_keys = new_keys - existing_keys
    if to_add_keys:
        rows_to_add = df_new[
            df_new.apply(lambda row: (row["CompetitorNumber"], row["LapNumber"]) in to_add_keys, axis=1)
        ]
        st.session_state['all_laps_df'] = pd.concat([existing, rows_to_add], ignore_index=True)
else:
    st.session_state['all_laps_df'] = df_new

df_all_laps = st.session_state['all_laps_df']

# === Sidebar Settings ===
st.sidebar.header("Settings")
threshold_sec = st.sidebar.number_input("Pitstop Threshold (seconds)", min_value=10.0, max_value=300.0, value=80.0, step=1.0)
threshold_ms = threshold_sec * 1000

all_kart_numbers = sorted(df_all_laps["CompetitorNumber"].dropna().unique())
selected_karts = st.sidebar.multiselect("Select Kart Number(s)", options=all_kart_numbers, default=[all_kart_numbers[0]] if all_kart_numbers else [])

# Mapping display names to internal values
y_axis_options = {
    "Lap Time": "LapTime",
    "Gap": "Gap",
    "Position": "Position",
    "Behind": "Behind"
}

# Use the keys (display names) in the selectbox
selected_label = st.sidebar.selectbox("Y-axis Variable", options=list(y_axis_options.keys()))

# Get the internal value for your logic
y_var = y_axis_options[selected_label]

x_lim_checkbox = st.sidebar.checkbox("Set X-axis limit")

# New checkbox to set max x axis automatically to current max lap of selected teams
x_axis_max_current_lap_checkbox = st.sidebar.checkbox("Set X-axis max to current lap")

x_axis_min_lap = 0
x_axis_max_lap_option = None

if x_lim_checkbox:
    x_axis_min_lap = st.sidebar.number_input("Min Lap for X-axis", min_value=0, max_value=1000, value=0, step=1)

    if x_axis_max_current_lap_checkbox and selected_karts:
        max_lap_default = int(df_all_laps[df_all_laps["CompetitorNumber"].isin(selected_karts)]["LapNumber"].max())
        x_axis_max_lap_option = max_lap_default
        # Show max lap info
        st.sidebar.markdown(f"**Max Lap (auto): {max_lap_default}**")
    else:
        max_lap_default = 100
        if selected_karts:
            max_lap_default = int(df_all_laps[df_all_laps["CompetitorNumber"].isin(selected_karts)]["LapNumber"].max())
        x_axis_max_lap_option = st.sidebar.number_input("Max Lap for X-axis", min_value=1, max_value=2000, value=max_lap_default, step=1)


y_lim_checkbox = st.sidebar.checkbox("Set Y-axis limit")
y_axis_min = None
y_axis_max = None
if y_lim_checkbox:
    y_axis_min = st.sidebar.number_input("Y-axis Min", value=0.0, step=0.1)
    y_axis_max = st.sidebar.number_input("Y-axis Max", value=100.0, step=0.1)

n_lap_avg = st.sidebar.number_input("Number of laps to average", min_value=1, max_value=2000, value=5, step=1)


# === Compute overall longest stint and fastest pitstop from all karts (all data) ===
longest_stint = {"kart": None, "laps": 0, "time": 0}
fastest_pitstop = {"kart": None, "lap_time_sec": None, "lap_number": None}
fastest_avg = {"kart": None, "avg_time": None}

for kart, df_kart in df_all_laps.groupby("CompetitorNumber"):
    if df_kart.empty:
        continue

    df_kart = df_kart.sort_values("LapNumber").reset_index(drop=True)
    df_kart["LapTime"] = pd.to_numeric(df_kart["LapTime"], errors="coerce")
    df_kart["IsPitStop"] = df_kart["LapTime"] > threshold_ms

    if len(df_kart) >= n_lap_avg:
        lap_window = df_kart.tail(n_lap_avg)["LapTime"].values
        if not np.all(np.isnan(lap_window)):
            rolling_median = np.nanmedian(lap_window) / 1000.0

        else:
            rolling_median = np.nan  # or skip this kart
            
        min_median = rolling_median

        if pd.notna(min_median):
            if (fastest_avg["avg_time"] is None) or (min_median < fastest_avg["avg_time"]):
                fastest_avg["kart"] = kart
                fastest_avg["avg_time"] = round(min_median, 3)

    # Longest stint calculation
    df_kart["StintNumber"] = df_kart["IsPitStop"].cumsum()
    for stint_num, group in df_kart.groupby("StintNumber"):
        race_laps = group[~group["IsPitStop"]]
        if race_laps.empty:
            continue

        stint_len = len(race_laps)
        stint_time_min = race_laps["LapTime"].sum() / 60000

        if stint_len > longest_stint["laps"]:
            longest_stint = {
                "kart": kart,
                "laps": stint_len,
                "time": round(stint_time_min, 2),
            }

    # Fastest pitstop detection
    pit_laps = df_kart[df_kart["IsPitStop"]]
    if not pit_laps.empty:
        fastest_row = pit_laps.loc[pit_laps["LapTime"].idxmin()]
        pit_time_sec = fastest_row["LapTime"] / 1000.0
        if (fastest_pitstop["lap_time_sec"] is None) or (pit_time_sec < fastest_pitstop["lap_time_sec"]):
            fastest_pitstop = {
                "kart": kart,
                "lap_time_sec": round(pit_time_sec, 3),
                "lap_number": int(fastest_row["LapNumber"])
            }


# === Exclude non-pit laps selection ===
# Find all laps above threshold that are NOT pit laps for selected karts
exclude_options = []
for kart in selected_karts:
    df_kart = df_all_laps[df_all_laps["CompetitorNumber"] == kart]
    laps_above_thresh = df_kart[df_kart["LapTime"] > threshold_ms]
    for _, row in laps_above_thresh.iterrows():
        lap_num = int(row['LapNumber'])
        lap_time_sec = row['LapTime'] / 1000.0
        exclude_options.append(f"Kart {kart}: Lap {lap_num} ({lap_time_sec:.2f}s)")
        
        

exclude_selected = st.sidebar.multiselect("Exclude non pit laps", options=exclude_options)

st.markdown("---")

refresh_rate = st.sidebar.number_input(
        "Refresh rate (seconds)",
        min_value=1,
        max_value=300,
        value=30,
        step=1,
        help="Set how often the dashboard refreshes automatically."
    )

# Auto-refresh every 30s
REFRESH_INTERVAL_MS = refresh_rate*1000
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="datarefresh")

# Map exclude_selected back to kart and lap number
exclude_laps = []
for val in exclude_selected:
    # val example: "Kart 12: Lap 23 (87.45s)"
    parts = val.split(":")
    if len(parts) == 2:
        kart_num = parts[0].replace("Kart ", "").strip()
        lap_part = parts[1].strip()  # e.g. "Lap 23 (87.45s)"
        # Extract lap number by splitting on space and taking second token
        lap_num_str = lap_part.split(" ")[1]
        lap_num = int(lap_num_str)
        exclude_laps.append( (kart_num, lap_num) )


# === Process selected karts for plotting and stint summaries ===

stint_tables = {}

fig, ax = plt.subplots(figsize=(10, 5))

for kart in selected_karts:
    df_laps = df_all_laps[df_all_laps["CompetitorNumber"] == kart].copy()
    if df_laps.empty:
        st.warning(f"No lap data for Kart {kart}.")
        continue

    df_laps["LapNumber"] = pd.to_numeric(df_laps["LapNumber"], errors='coerce')
    df_laps["LapTime"] = pd.to_numeric(df_laps["LapTime"], errors='coerce')

    # Identify pit laps after excluding selected non-pit laps
    # Mark laps above threshold as pit laps unless in exclude_laps list
    def is_pit(row):
        lap_time = row["LapTime"]
        # Check if lap_time is missing or NaN: treat as NOT pit stop
        if pd.isna(lap_time):
            return False
        # Now check threshold
        if lap_time <= threshold_ms:
            return False
        # Check exclude laps as you currently do
        if (row["CompetitorNumber"], row["LapNumber"]) in exclude_laps:
            return False
        return True

    df_laps["IsPitStop"] = df_laps.apply(is_pit, axis=1)

    # Set stint number
    df_laps["StintNumber"] = df_laps["IsPitStop"].cumsum()

    # Convert Gap and Behind to numeric if possible
    for col in ["Gap", "Behind"]:
        if col in df_laps.columns:
            df_laps[col] = pd.to_numeric(df_laps[col], errors='coerce')

    # Plotting
    if y_var not in df_laps.columns:
        st.warning(f"{y_var} not found for Kart {kart}, skipping plot.")
        continue

    if y_var == "LapTime":
        y_values = df_laps["LapTime"] / 1000.0  # convert ms to s
        y_label = "Lap Time (s)"
    else:
        y_values = df_laps[y_var]
        y_label = y_var

    ax.plot(df_laps["LapNumber"], y_values, label=f"Kart {kart}")

    # Stint summaries for table & title info
    stint_summary = []
    for stint_num, group in df_laps.groupby("StintNumber"):
        race_laps = group[~group["IsPitStop"]]
        if race_laps.empty:
            continue

        start_lap = int(race_laps["LapNumber"].min())
        end_lap = int(race_laps["LapNumber"].max())
        stint_len = len(race_laps)
        stint_time_min = race_laps["LapTime"].sum() / 60000

        best_lap_sec = race_laps["LapTime"].min() / 1000
        median_lap_sec = np.nanmedian(race_laps["LapTime"].values) / 1000

        # Get pit lap time at the end of the stint, if any pit lap in group
        pit_lap_time_ms = None
        pit_laps_in_group = group[group["IsPitStop"]]
        if not pit_laps_in_group.empty:
            # Prefer the pit lap immediately after the stint end lap
            next_pit_lap = pit_laps_in_group[pit_laps_in_group["LapNumber"] == (end_lap + 1)]
            if not next_pit_lap.empty:
                pit_lap_time_ms = next_pit_lap["LapTime"].iloc[0]
            else:
                # Otherwise take the last pit lap in this group
                pit_lap_time_ms = pit_laps_in_group["LapTime"].iloc[-1]

        pit_lap_time_sec = round(pit_lap_time_ms / 1000, 3) if pit_lap_time_ms is not None else None

        stint_summary.append({
            "Stint No.": stint_num + 1,  # shift numbering to start from 1
            "Start Lap": start_lap,
            "End Lap": end_lap,
            "Stint Laps": stint_len,
            "Stint Time(min)": round(stint_time_min, 2),
            "Best Lap (s)": round(best_lap_sec, 3),
            "Median Lap (s)": round(median_lap_sec, 3),
            "Pit Time (s)": pit_lap_time_sec
        })

    df_summary = pd.DataFrame(stint_summary)
    
    if "Pit Time (s)" in df_summary.columns:
        df_summary["Pit Time (s)"] = df_summary["Pit Time (s)"].shift(-1)   


    # Format seconds columns to strings without trailing zeros
    for col in ["Best Lap (s)", "Median Lap (s)", "Pit Time (s)"]:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(format_seconds)
    stint_tables[kart] = df_summary

    # Last pit lap number and n lap avg lap time for title info
    pit_laps_all = df_laps[df_laps["IsPitStop"]]
    last_pit_lap = None
    if not pit_laps_all.empty:
        last_pit_lap = int(pit_laps_all["LapNumber"].max())

    # n lap avg
    last_n_avg_sec = None
    delta_str = ""
    if len(df_laps) >= n_lap_avg:
        last_n_avg = np.nanmedian(df_laps.tail(n_lap_avg)["LapTime"].values) / 1000.0
        last_n_avg_sec = round(last_n_avg, 3)

        # Compute delta to fastest average
        if fastest_avg.get("avg_time") is not None:
            delta = last_n_avg - fastest_avg["avg_time"]
            delta_str = f" (+{delta:.3f})"

    # Title output
    avg_str = f"{last_n_avg_sec:.3f} {delta_str}" if last_n_avg_sec is not None else "N/A"
    st.write(f"### Kart {kart} — Last pit: {last_pit_lap if last_pit_lap is not None else 'N/A'}, {n_lap_avg} lap avg: {avg_str}")
    st.dataframe(df_summary, hide_index=True)
    
# Finalize plot axes
ax.set_xlabel("Lap Number")
ax.set_ylabel(y_label)
ax.set_title("Live Timing Analysis")
ax.legend()
ax.grid(True)
if x_lim_checkbox:
    ax.set_xlim(left=x_axis_min_lap, right=x_axis_max_lap_option)
if y_lim_checkbox:
    ax.set_ylim(bottom=y_axis_min, top=y_axis_max)

st.pyplot(fig)


# Refresh info footer
st.markdown("---")
if fastest_avg["kart"]:
    st.markdown(
        f"**Fastest {n_lap_avg}-lap average:** Kart {fastest_avg['kart']} "
        f"({fastest_avg['avg_time']} sec)"
    )
else:
    st.markdown(f"**Fastest {n_lap_avg}-lap average:** Not enough data yet.")
    
st.write(f"**Longest stint:** Kart {longest_stint['kart']} - {longest_stint['laps']} laps, {longest_stint['time']} min")
st.write(f"**Fastest pitstop:** Kart {fastest_pitstop['kart']} - {fastest_pitstop['lap_time_sec']} sec (Lap {fastest_pitstop['lap_number']})")
st.markdown("---")
st.markdown(f"⏳ Refreshing every {refresh_rate} seconds.")
local_time = datetime.now(ZoneInfo("Europe/London"))
st.write(f"*Last refresh: {local_time.strftime('%Y-%m-%d %H:%M:%S')}*")
st.write("🏁 MS URBN BLUE are the best 🏁")
