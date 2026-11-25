"""JalMarg demo: pick a vehicle and a moment in the storm; see depths, forecasts and advice.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jalmarg.demo.scenario import DURATION_MIN, RAIN_STOP_MIN, START, replay, true_depths  # noqa: E402
from jalmarg.domain.vehicles import VehicleProfileFactory  # noqa: E402
from jalmarg.routing.planner import DISCLAIMER  # noqa: E402
from jalmarg.temporal.forecaster import PhysicsForecaster, default_forecaster  # noqa: E402

st.set_page_config(page_title="JalMarg demo", layout="wide")
st.title("JalMarg: wait or detour?")
st.caption("Recorded storm replayed as a live stream. Demo area and roads are illustrative.")

factory = VehicleProfileFactory()
vehicles = {v.label: v.key for v in factory.all()}

with st.sidebar:
    vehicle_label = st.selectbox("Your vehicle", list(vehicles), index=1)
    vehicle_key = vehicles[vehicle_label]
    minute = st.slider("Minutes since rain started", 0, DURATION_MIN - 61, 40)
    model = st.radio("Forecaster", ["Hybrid (physics + GRU)", "Physics only"])
    st.markdown(f"Rain: **{'40 mm/h' if minute < RAIN_STOP_MIN else 'stopped'}**")
    st.warning(DISCLAIMER)


@st.cache_resource
def get_forecaster(name: str):
    return PhysicsForecaster() if name == "Physics only" else default_forecaster()


@st.cache_resource(max_entries=64)
def get_service(minute: int, model_name: str):
    return replay(minute, forecaster=get_forecaster(model_name))


svc = get_service(minute, model)
t0 = START + timedelta(minutes=minute)
advice = svc.plan_route(vehicle_key, "home", "office", t0)
vehicle = factory.create(vehicle_key)

# --- advice ---------------------------------------------------------------
box = {"go_now": st.success, "wait": st.info, "no_safe_route": st.error}[advice.decision]
box(f"**{advice.message}**  \nRisk: {advice.risk}. Time {t0:%H:%M}.")
if advice.naive_unsafe_segments:
    st.caption("A water-blind router would send you through: "
               + ", ".join(svc.segments.get(s).name for s in advice.naive_unsafe_segments))

col_map, col_seg = st.columns([3, 2])

# --- map ------------------------------------------------------------------
with col_map:
    g = svc.graph
    m = folium.Map(location=[30.327, 78.015], zoom_start=14, tiles="OpenStreetMap")
    seg_color = {}
    for s in svc.segments.all():
        ttp = s.time_to_passable(vehicle, t0)
        seg_color[s.segment_id] = {"now": "green", "in": "orange"}.get(ttp.status, "red")
    for u, v, d in g.G.edges(data=True):
        if str(u) > str(v):
            continue
        sid = d.get("segment_id")
        folium.PolyLine([g.coords(u), g.coords(v)], color=seg_color.get(sid, "#888"),
                        weight=7 if sid else 4, opacity=0.8, tooltip=d.get("name")).add_to(m)
    if advice.recommended:
        folium.PolyLine([g.coords(n) for n in advice.recommended.nodes], color="#1f4fff",
                        weight=3, dash_array="8", tooltip="Recommended").add_to(m)
    for n in ("home", "office"):
        folium.Marker(g.coords(n), tooltip=g.name(n)).add_to(m)
    st_folium(m, height=460, use_container_width=True, returned_objects=[])
    st.caption("Segment colour for your vehicle: green passable now, orange opens within the hour, "
               "red not expected within the hour or unknown. Blue dashes: recommended route.")

# --- segments -------------------------------------------------------------
with col_seg:
    rows = []
    for s in svc.segments.all():
        ttp = s.time_to_passable(vehicle, t0)
        rows.append({"Segment": s.name, "State": s.state.name, "Depth (cm)": round(s.depth_cm, 1),
                     "± (cm)": round(s.sigma_cm, 1), "For you": ttp.describe(vehicle.label)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    sel = st.selectbox("Forecast for", [s.name for s in svc.segments.all()])
    seg = next(s for s in svc.segments.all() if s.name == sel)
    q = seg.forecast(t0)
    truth = true_depths()[seg.segment_id]
    if q is not None:
        hist_t = np.arange(max(0, minute - 60), minute)
        fut_t = minute + q.minutes.astype(int)
        df = pd.DataFrame({"minute": np.concatenate([hist_t, fut_t])}).set_index("minute")
        df["true depth"] = np.interp(df.index, np.arange(len(truth)), truth)
        df["forecast q50"] = np.nan
        df["q10"] = np.nan
        df["q90 (used for decisions)"] = np.nan
        df.loc[fut_t, "forecast q50"] = q.q50
        df.loc[fut_t, "q10"] = q.q10
        df.loc[fut_t, "q90 (used for decisions)"] = q.q90
        df[f"{vehicle.label} limit"] = vehicle.threshold_cm
        st.line_chart(df)

with st.expander("Alerts (municipal view)"):
    for a in svc.alerts.alerts[-15:][::-1]:
        st.write(f"{a.time:%H:%M} · {a.message}")
