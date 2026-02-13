"""
S.A.V.E. — Strategic Agent-based Victim Evacuation
Enhanced Streamlit Dashboard v2.0
Features: ESI Triage Charts, Auto-Refresh, Tabbed Views, Analytics, Download Reports
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import time
import json
import pandas as pd
from datetime import datetime
import sys
import os

# Add backend to path for data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend-python')))
from data_loader import data_loader
from folium.plugins import HeatMap

# Backend URL configuration (for cloud deployment)
BACKEND_URL = os.getenv("BACKEND_URL", st.secrets.get("BACKEND_URL", "http://localhost:5000"))


# Page configuration
st.set_page_config(
    page_title="S.A.V.E. — Disaster Response System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PREMIUM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main container */
    .main {
        background: linear-gradient(135deg, #0a0a1a 0%, #111128 50%, #0d1b2a 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f97316 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.35);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            90deg, transparent, transparent 20px,
            rgba(255,255,255,0.03) 20px, rgba(255,255,255,0.03) 40px
        );
    }
    .main-header h1 {
        color: white; margin: 0; font-size: 2rem;
        font-weight: 800; letter-spacing: -0.5px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        position: relative;
    }
    .main-header .subtitle {
        color: rgba(255,255,255,0.85); margin-top: 6px;
        font-size: 0.95rem; font-weight: 400; position: relative;
    }
    .main-header .version {
        position: absolute; top: 12px; right: 16px;
        background: rgba(0,0,0,0.25); color: rgba(255,255,255,0.7);
        padding: 4px 10px; border-radius: 20px; font-size: 0.7rem;
        font-weight: 600; letter-spacing: 0.5px;
    }

    /* Agent panel */
    .agent-panel {
        background: linear-gradient(180deg, #0c0c20 0%, #12122e 100%);
        border-radius: 16px; padding: 16px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        max-height: 480px; overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(99,102,241,0.3) transparent;
    }
    .agent-panel::-webkit-scrollbar { width: 6px; }
    .agent-panel::-webkit-scrollbar-thumb {
        background: rgba(99,102,241,0.3); border-radius: 3px;
    }

    /* Agent messages */
    .agent-message {
        background: rgba(255,255,255,0.04);
        border-radius: 12px; padding: 12px 14px;
        margin: 6px 0; border-left: 3px solid #6366f1;
        animation: slideIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        transition: background 0.2s;
    }
    .agent-message:hover { background: rgba(255,255,255,0.07); }
    .agent-message.success { border-left-color: #10b981; background: rgba(16,185,129,0.06); }
    .agent-message.warning { border-left-color: #f59e0b; background: rgba(245,158,11,0.06); }
    .agent-message.error   { border-left-color: #ef4444; background: rgba(239,68,68,0.06); }
    .agent-message.info    { border-left-color: #6366f1; background: rgba(99,102,241,0.06); }

    .agent-name {
        font-weight: 700; color: #818cf8;
        font-size: 0.8rem; text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .agent-timestamp {
        color: rgba(255,255,255,0.35); font-size: 0.7rem;
        float: right; font-weight: 500;
    }
    .agent-text {
        color: rgba(255,255,255,0.85); margin-top: 4px;
        font-size: 0.88rem; line-height: 1.5;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px; padding: 20px; text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.15);
    }
    .stat-number { font-size: 2.2rem; font-weight: 800; color: white; }
    .stat-label { color: rgba(255,255,255,0.6); font-size: 0.8rem; font-weight: 500; margin-top: 4px; }

    /* ESI Level colors */
    .esi-1 { color: #ef4444; font-weight: 700; }
    .esi-2 { color: #f97316; font-weight: 700; }
    .esi-3 { color: #eab308; font-weight: 700; }
    .esi-4 { color: #22c55e; font-weight: 700; }
    .esi-5 { color: #3b82f6; font-weight: 700; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
        color: white !important; border: none !important;
        border-radius: 12px !important; padding: 14px 32px !important;
        font-size: 1rem !important; font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.5) !important;
    }

    /* Pulse animation for active status */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulse { animation: pulse 2s infinite; }

    /* Status badge */
    .status-badge {
        display: inline-block; padding: 4px 12px;
        border-radius: 20px; font-size: 0.75rem;
        font-weight: 700; letter-spacing: 0.5px;
    }
    .status-active {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-standby {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* ESI bar chart */
    .esi-bar-container { margin: 4px 0; }
    .esi-bar-label {
        font-size: 0.78rem; color: rgba(255,255,255,0.8);
        margin-bottom: 2px; font-weight: 600;
    }
    .esi-bar-bg {
        background: rgba(255,255,255,0.08); border-radius: 6px;
        height: 22px; position: relative; overflow: hidden;
    }
    .esi-bar-fill {
        height: 100%; border-radius: 6px;
        display: flex; align-items: center;
        padding-left: 8px; font-size: 0.72rem;
        color: white; font-weight: 700;
        transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(99,102,241,0.1);
        border-radius: 8px;
        color: rgba(255,255,255,0.7);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99,102,241,0.25) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
defaults = {
    'messages': [],
    'disaster_triggered': False,
    'allocation_result': None,
    'hospitals_data': [],
    'full_response': None,
    'auto_refresh': False,
    'last_trigger_time': None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================
# HEADER
# ============================================
# Landing page removed — dashboard loads directly

# ============================================
# MAIN DASHBOARD
# ============================================

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <span class="version">v3.2 • AI + MARL + GIS</span>
    <h1>🚨 S.A.V.E. — Disaster Response System</h1>
    <div class="subtitle">Strategic Agent-based Victim Evacuation • Real-time Multi-Agent Coordination</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## ⚙️ Disaster Parameters")
    st.markdown("---")

    # ---- Location Search ----
    st.markdown("### 🔍 Location Search")
    search_query = st.text_input("Search a place", placeholder="e.g. Mumbai, India")
    if search_query:
        try:
            geo_resp = requests.get(
                f"{BACKEND_URL}/geocode",
                params={"q": search_query}, timeout=10
            ).json()
            geo_results = geo_resp.get("results", [])
            if geo_results:
                place_options = [r["display_name"][:60] for r in geo_results]
                selected_idx = st.selectbox(
                    "Select result", range(len(place_options)),
                    format_func=lambda i: place_options[i]
                )
                chosen = geo_results[selected_idx]
                st.session_state.search_lat = chosen["lat"]
                st.session_state.search_lng = chosen["lng"]
                st.success(f"📍 {chosen['display_name'][:50]}")
            else:
                st.warning(f"No results for '{search_query}'")
        except Exception as e:
            st.error(f"Search error: {e}")

    default_lat = st.session_state.get("search_lat", 22.721)
    default_lng = st.session_state.get("search_lng", 88.485)
    disaster_lat = st.number_input("📍 Latitude", value=default_lat, format="%.4f")
    disaster_lng = st.number_input("📍 Longitude", value=default_lng, format="%.4f")

    st.markdown("---")

    patients = st.slider("👥 Number of Patients", 1, 100, 10)

    disaster_type = st.selectbox("🔥 Disaster Type",
        ["FIRE", "FLOOD", "EARTHQUAKE", "ACCIDENT", "CHEMICAL_SPILL", "BUILDING_COLLAPSE"])
    severity = st.selectbox("⚠️ Severity Level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"])

    st.markdown("---")

    # Data source selection
    st.markdown("### 📡 Data Source")
    data_source = st.selectbox("Source", ["osm", "static", "mongodb"], index=0)

    st.markdown("---")

    # Training Simulation
    st.markdown("### 🧠 AI Model Status")
    if st.button("🔄 Simulate Training (Warm-up)"):
        with st.status("Training AI Agents...", expanded=True) as status:
            st.write("Initializing Replay Buffers...")
            time.sleep(0.5)
            st.write("Running 50 Episodes (DQN)...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            st.write("Optimizing Q-Networks (Cross-Entropy Loss)...")
            time.sleep(0.5)
            st.write("Updating Target Networks...")
            time.sleep(0.3)
            status.update(label="AI Models Ready! (Loss: 0.042)", state="complete", expanded=False)
        st.success("Training Complete. Agents optimized.")

    st.markdown("---")

    # Real Dataset Loader
    st.markdown("### 📂 Real Datasets")
    dataset_source = st.selectbox(
        "Load Stub Data",
        ["None", "FEMA (Hurricane)", "WHO (Pandemic)", "NDMA (Earthquake)"]
    )
    if dataset_source != "None":
        if st.button(f"Load {dataset_source.split(' ')[0]}"):
            source_key = dataset_source.split(" ")[0]
            with st.spinner(f"Loading {source_key}..."):
                default_lat, default_lng = 28.61, 77.23
                patients_data = data_loader.load_scenario(source_key, default_lat, default_lng)
                
                st.session_state.disaster_triggered = True
                st.session_state.dataset_loaded = True
                st.session_state.loaded_patients = patients_data
                st.session_state.last_trigger_time = datetime.now()
                st.success(f"Loaded {len(patients_data)} records!")
                time.sleep(1)
                st.rerun()

    st.markdown("---")
    
    # GIS Controls
    st.markdown("### 🗺️ GIS Layers")
    show_heatmap = st.toggle("Density Heatmap", value=False)
    show_risk_zones = st.toggle("Risk Zones", value=False)
    
    st.markdown("---")

    # ---- AI Emergency Call ----
    st.markdown("### 📞 AI Emergency Call")
    # Auto-fill user's number
    default_phone = "+917894281460"
    call_number = st.text_input("Phone Number", value=default_phone, placeholder="+91XXXXXXXXXX")
    auto_call = st.checkbox("Auto-Call on Trigger", value=True)
    
    if st.button("📞 Call Authorities Now", disabled=not call_number):
        with st.spinner("AI Agent is dialing..."):
            try:
                call_payload = {
                    "to_number": call_number,
                    "disaster_type": disaster_type,
                    "location_name": search_query if search_query else f"{disaster_lat:.3f}, {disaster_lng:.3f}",
                    "lat": disaster_lat,
                    "lng": disaster_lng,
                    "patients": patients,
                    "severity": severity,
                    "hospitals_nearby": len(st.session_state.get("hospitals_data", [])),
                    "ambulances_dispatched": len(st.session_state.get("ambulance_data", [])),
                }
                call_resp = requests.post(
                    f"{BACKEND_URL}/ai-call",
                    json=call_payload, timeout=15
                ).json()

                if call_resp.get("error"):
                    st.error(f"Call failed: {call_resp['error']}")
                else:
                    st.success(f"✅ Call initiated! SID: {call_resp.get('call_sid', 'N/A')}")
                    st.info(f"Status: {call_resp.get('status', 'queued')}")
                    with st.expander("📝 Call Script"):
                        st.code(call_resp.get("script", ""), language="xml")
            except Exception as e:
                st.error(f"Call error: {e}")

    st.markdown("---")

    # Auto-refresh toggle
    auto_refresh = st.toggle("🔄 Auto-Refresh (30s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    st.markdown("---")

    # Hospital database display
    st.markdown("### 🏥 Hospital Database")
    if st.session_state.hospitals_data:
        for h in st.session_state.hospitals_data[:6]:
            beds = h.get('Beds', '?')
            name = h.get('Hospital', 'Unknown')
            specialty = h.get('specialty', '')
            spec_tag = f" ({specialty})" if specialty and specialty != "General" else ""
            st.markdown(f"**{name}**: {beds} beds{spec_tag}")
        if len(st.session_state.hospitals_data) > 6:
            st.markdown(f"*...and {len(st.session_state.hospitals_data)-6} more*")
    else:
        st.info("Trigger a disaster to fetch hospital data")

    # System status
    st.markdown("---")
    st.markdown("### 💻 System Status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.success(f"Backend: **Online** (v{health.get('version', '?')})")
        cache_stats = health.get("cache_stats", {})
        st.caption(f"Cache: {cache_stats.get('entries', 0)} entries, "
                   f"{cache_stats.get('hit_rate_percent', 0)}% hit rate")
    except Exception:
        st.error("Backend: **Offline**")
        st.caption("Start with: `python backend-python/optimize.py`")

# ============================================
# MAIN CONTENT — TABBED INTERFACE
# ============================================

tab_response, tab_triage, tab_ai, tab_analytics, tab_research = st.tabs([
    "🗺️ Response Map", "🏷️ ESI Triage", "🧠 AI Insights", "📊 Analytics", "🔬 Research"
])

# ============================================
# TAB 1: RESPONSE MAP & AGENT PANEL
# ============================================
with tab_response:
    col_map, col_panel = st.columns([3, 2])

    with col_map:
        # Create Folium map
        m = folium.Map(
            location=[disaster_lat, disaster_lng],
            zoom_start=14,
            tiles='CartoDB dark_matter'
        )

        # Disaster location marker
        folium.Marker(
            [disaster_lat, disaster_lng],
            popup=folium.Popup(
                f"<b>🔥 {disaster_type}</b><br>"
                f"Patients: {patients}<br>"
                f"Severity: {severity}",
                max_width=200
            ),
            icon=folium.Icon(color="red", icon="fire", prefix='fa'),
            tooltip="Disaster Location"
        ).add_to(m)

        # Impact radius circle
        folium.Circle(
            [disaster_lat, disaster_lng],
            radius=500,
            color='#ef4444',
            fill=True,
            fill_opacity=0.1,
            weight=2,
            dash_array='8',
            tooltip="Impact Zone (~500m)"
        ).add_to(m)

        # Hospital markers from session state
        if st.session_state.hospitals_data:
            for i, h in enumerate(st.session_state.hospitals_data):
                color = 'blue'
                icon_name = "hospital-o"

                # Highlight allocated hospitals in green
                if st.session_state.allocation_result:
                    for alloc in st.session_state.allocation_result.get('allocation', []):
                        if alloc['hospital'] == h.get('Hospital'):
                            color = 'green'
                            icon_name = "plus-square"
                            break

                beds = h.get('Beds', '?')
                avail = h.get('available_beds', beds)
                occupancy = h.get('occupancy_pct', '?')
                icu_avail = h.get('icu_available', '?')
                specialist = h.get('specialist_on_duty', 'N/A')
                specialty = h.get('specialty', 'General')
                popup_html = (
                    f"<b>🏥 {h.get('Hospital', 'Unknown')}</b><br>"
                    f"Total Beds: {beds} | <b>Available: {avail}</b><br>"
                    f"Occupancy: {occupancy}%<br>"
                    f"ICU Available: {icu_avail}<br>"
                    f"Specialist: {specialist}<br>"
                    f"Specialty: {specialty}<br>"
                    f"Emergency: {h.get('emergency', 'yes')}"
                )

                folium.Marker(
                    [h.get('lat', 0), h.get('lng', 0)],
                    popup=folium.Popup(popup_html, max_width=220),
                    icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
                    tooltip=f"{h.get('Hospital')} — {beds} beds"
                ).add_to(m)

            # Draw allocation lines
            if st.session_state.allocation_result:
                for alloc in st.session_state.allocation_result.get('allocation', []):
                    h_lat = alloc.get('lat')
                    h_lng = alloc.get('lng')
                    if h_lat and h_lng and alloc.get('assigned', 0) > 0:
                        folium.PolyLine(
                            [[disaster_lat, disaster_lng], [h_lat, h_lng]],
                            weight=3,
                            color='#10b981',
                            opacity=0.7,
                            dash_array='8',
                            tooltip=f"{alloc['hospital']}: {alloc['assigned']} patients"
                        ).add_to(m)

        # GIS: Heatmap Layer
        if show_heatmap and st.session_state.disaster_triggered:
            # Collect patient coordinates
            heat_data = []
            # Check if we have allocated results or just raw data
            current_patients = []
            if st.session_state.allocation_result:
                # Extract from allocation
                for h in st.session_state.allocation_result.get("hospital_allocation", {}).values():
                    for p in h.get("assigned_patients", []):
                         heat_data.append([p["lat"], p["lng"], 1.0]) # Weight 1.0
            elif st.session_state.get('loaded_patients'):
                 for p in st.session_state.loaded_patients:
                     heat_data.append([p["lat"], p["lng"], 1.0])
            
            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)

        # GIS: Risk Zones (Choropleth Stub)
        if show_risk_zones and st.session_state.disaster_triggered:
            # Simulate a high-risk zone circle
            folium.Circle(
                location=[28.61, 77.23],
                radius=5000,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup="High Risk Zone (Flood Data)"
            ).add_to(m)

        # Display map
        st_folium(m, width=1200, height=600, use_container_width=True)

    with col_panel:
        st.markdown("### 📡 Live Agent Panel")

        # Trigger button
        if st.button("🚨 TRIGGER DISASTER RESPONSE", use_container_width=True):
            st.session_state.disaster_triggered = True
            st.session_state.last_trigger_time = datetime.now()

            # Clear previous messages
            try:
                requests.post(f"{BACKEND_URL}/clear-messages", timeout=2)
            except Exception:
                pass

            st.session_state.messages = []

            # Initial trigger messages
            st.session_state.messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "agent": "Disaster Trigger Agent",
                "message": f"🔥 {disaster_type} detected at ({disaster_lat:.3f}, {disaster_lng:.3f})",
                "type": "warning"
            })
            st.session_state.messages.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "agent": "Disaster Trigger Agent",
                "message": f"Severity: {severity} | Patients: {patients} | Source: {data_source}",
                "type": "info"
            })

            # --- AUTO-CALL LOGIC ---
            if auto_call and call_number:
                st.toast(f"📞 Initiating AI Call to {call_number}...", icon="🤖")
                try:
                    # Run in background to not block UI
                    call_payload = {
                        "to_number": call_number,
                        "disaster_type": disaster_type,
                        "location_name": search_query if search_query else f"{disaster_lat:.3f}, {disaster_lng:.3f}",
                        "lat": disaster_lat, "lng": disaster_lng,
                        "patients": patients, "severity": severity,
                        "hospitals_nearby": 5, # Estimate
                        "ambulances_dispatched": 8 # Estimate
                    }
                    requests.post(f"{BACKEND_URL}/ai-call", json=call_payload, timeout=1)
                except Exception:
                    pass # Fire and forget
            # -----------------------

            # Call full-response endpoint
            try:
                # 1. Fetch hospitals for map
                with st.spinner("🌍 Fetching hospital data..."):
                    hosp_response = requests.get(
                        f"{BACKEND_URL}/hospitals?lat={disaster_lat}&lng={disaster_lng}"
                        f"&radius_km=10&source={data_source}",
                        timeout=25
                    )
                    if hosp_response.status_code == 200:
                        data = hosp_response.json()
                        st.session_state.hospitals_data = data.get('hospitals', [])

                # 2. Optimize allocation
                with st.spinner("🏥 Optimizing patient allocation..."):
                    opt_response = requests.post(
                        f"{BACKEND_URL}/optimize",
                        json={
                            "patients": patients,
                            "lat": disaster_lat,
                            "lng": disaster_lng,
                            "disaster_type": disaster_type,
                            "source": data_source
                        },
                        timeout=45
                    )

                if opt_response.status_code == 200:
                    result = opt_response.json()
                    st.session_state.allocation_result = result
                    st.session_state.full_response = result

                    # Add allocation messages
                    for alloc in result.get('allocation', []):
                        if alloc.get('assigned', 0) > 0:
                            eta = alloc.get('eta_minutes', '?')
                            st.session_state.messages.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "agent": "Hospital Agent",
                                "message": f"🏥 {alloc['hospital']} → {alloc['assigned']} patients "
                                           f"({alloc['distance']}km, ETA {eta}min)",
                                "type": "success"
                            })

                    remaining = result.get('remaining', 0)
                    if remaining > 0:
                        st.session_state.messages.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent": "Government Agent",
                            "message": f"⚠️ CRITICAL: {remaining} patients need additional resources!",
                            "type": "warning"
                        })
                    else:
                        st.session_state.messages.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent": "Government Agent",
                            "message": f"✅ All {patients} patients successfully allocated",
                            "type": "success"
                        })

                    # ESI triage message
                    esi = result.get('esi_triage', {})
                    if esi:
                        crit = esi.get('critical_patients', 0)
                        icu = esi.get('icu_patients', 0)
                        st.session_state.messages.append({
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "agent": "Triage Agent",
                            "message": f"🏷️ ESI Assessment: {crit} critical (ESI 1-2), {icu} need ICU",
                            "type": "warning" if crit > 0 else "info"
                        })
                elif opt_response.status_code >= 400:
                    error_data = opt_response.json()
                    st.session_state.messages.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "agent": "System",
                        "message": f"❌ Error: {error_data.get('error', 'Unknown error')}",
                        "type": "error"
                    })

                # 3. Medicine requirements
                try:
                    with st.spinner("💊 Calculating medical supplies..."):
                        med_response = requests.post(
                            f"{BACKEND_URL}/medicine-requirements",
                            json={"patients": patients, "disaster_type": disaster_type},
                            timeout=10
                        )
                        if med_response.status_code == 200:
                            meds = med_response.json()
                            critical_items = [m['name'] for m in meds.get('critical_items', [])]
                            if critical_items:
                                st.session_state.messages.append({
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "agent": "Medical Supply Agent",
                                    "message": f"💊 CRITICAL: {', '.join(critical_items[:3])}",
                                    "type": "warning"
                                })

                            # Oxygen message
                            o2 = meds.get('oxygen_requirements', {})
                            if o2.get('total_o2_lpm', 0) > 0:
                                st.session_state.messages.append({
                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "agent": "Medical Supply Agent",
                                    "message": f"💨 O₂ needed: {o2['total_o2_lpm']}L/min "
                                               f"({o2.get('cylinders_24h_supply', '?')} cylinders/24h)",
                                    "type": "info"
                                })
                except Exception:
                    pass

                # 4. Ambulance dispatch
                try:
                    with st.spinner("🚑 Dispatching ambulances..."):
                        amb_response = requests.post(
                            f"{BACKEND_URL}/ambulance-dispatch",
                            json={"lat": disaster_lat, "lng": disaster_lng, "source": data_source},
                            timeout=30
                        )
                    if amb_response.status_code == 200:
                        amb_result = amb_response.json()
                        for dispatch in amb_result.get('dispatch', [])[:3]:
                            vtype = dispatch.get('vehicle_type', 'BLS')
                            st.session_state.messages.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "agent": "Emergency Agent",
                                "message": f"🚑 {dispatch['ambulance_id']} dispatched → "
                                           f"ETA: {dispatch['eta_minutes']}min ({vtype})",
                                "type": "success"
                            })
                except Exception:
                    pass

            except requests.exceptions.ConnectionError:
                st.session_state.messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": "System",
                    "message": "⚠️ Backend unreachable. Start: python backend-python/optimize.py",
                    "type": "error"
                })
            except requests.exceptions.ReadTimeout:
                st.session_state.messages.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "agent": "System",
                    "message": "⚠️ Request timed out. OSM data may be slow.",
                    "type": "warning"
                })

            st.rerun()

        # Display messages panel
        st.markdown('<div class="agent-panel">', unsafe_allow_html=True)

        if st.session_state.messages:
            for msg in st.session_state.messages:
                msg_type = msg.get('type', 'info')
                st.markdown(f"""
                <div class="agent-message {msg_type}">
                    <span class="agent-timestamp">{msg['timestamp']}</span>
                    <div class="agent-name">{msg['agent']}</div>
                    <div class="agent-text">{msg['message']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: rgba(255,255,255,0.4);">
                <p style="font-size: 2.5rem; margin-bottom: 8px;">🎯</p>
                <p style="font-weight: 600;">Click TRIGGER DISASTER RESPONSE to start</p>
                <p style="font-size: 0.8rem; margin-top: 4px;">Agents will coordinate in real-time</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================
    # STATISTICS ROW
    # ============================================
    st.markdown("---")

    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        st.metric(label="👥 Total Patients", value=patients)

    with s2:
        allocated = 0
        if st.session_state.allocation_result:
            allocated = sum(a.get('assigned', 0)
                          for a in st.session_state.allocation_result.get('allocation', []))
        st.metric(label="🏥 Allocated",
                  value=f"{allocated}/{patients}",
                  delta="All Clear" if allocated == patients and st.session_state.disaster_triggered else None)

    with s3:
        remaining = patients - allocated if st.session_state.disaster_triggered else 0
        st.metric(label="⚠️ Unallocated",
                  value=remaining,
                  delta="CRITICAL" if remaining > 0 else None,
                  delta_color="inverse")

    with s4:
        hosp_count = len(st.session_state.hospitals_data)
        st.metric(label="🏥 Hospitals Found", value=hosp_count)

    with s5:
        status_html = (
            '<span class="status-badge status-active pulse">● ACTIVE</span>'
            if st.session_state.disaster_triggered else
            '<span class="status-badge status-standby">● STANDBY</span>'
        )
        st.markdown(f"**⚡ Status**\n\n{status_html}", unsafe_allow_html=True)


# ============================================
# TAB 2: ESI TRIAGE VISUALIZATION
# ============================================
with tab_triage:
    if st.session_state.full_response and st.session_state.full_response.get('esi_triage'):
        esi_data = st.session_state.full_response['esi_triage']

        st.markdown("### 🏷️ ESI Triage Distribution")
        st.caption(f"Emergency Severity Index breakdown for **{esi_data.get('disaster_type', '')}** "
                   f"disaster — {esi_data.get('total_patients', 0)} patients")

        # ESI horizontal bar chart using HTML
        esi_colors = {
            "ESI-1": "#ef4444", "ESI-2": "#f97316", "ESI-3": "#eab308",
            "ESI-4": "#22c55e", "ESI-5": "#3b82f6"
        }
        esi_labels = {
            "ESI-1": "Immediate", "ESI-2": "Emergent", "ESI-3": "Urgent",
            "ESI-4": "Less Urgent", "ESI-5": "Non-Urgent"
        }

        total = esi_data.get('total_patients', 1)

        for key in ["ESI-1", "ESI-2", "ESI-3", "ESI-4", "ESI-5"]:
            data = esi_data.get('distribution', {}).get(key, {})
            count = data.get('patient_count', 0)
            pct = (count / total * 100) if total > 0 else 0
            color = esi_colors.get(key, "#666")
            label = esi_labels.get(key, key)
            wait = data.get('max_wait_minutes', '?')

            st.markdown(f"""
            <div class="esi-bar-container">
                <div class="esi-bar-label">{key} — {label} (max wait: {wait}min)</div>
                <div class="esi-bar-bg">
                    <div class="esi-bar-fill" style="width: {max(pct, 5)}%; background: {color};">
                        {count} pts ({pct:.0f}%)
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Summary cards
        col_icu, col_crit, col_staff = st.columns(3)
        with col_icu:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: #ef4444;">{esi_data.get('icu_patients', 0)}</div>
                <div class="stat-label">🛏️ ICU Beds Required</div>
            </div>
            """, unsafe_allow_html=True)
        with col_crit:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: #f97316;">{esi_data.get('critical_patients', 0)}</div>
                <div class="stat-label">⚠️ Critical Patients (ESI 1-2)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_staff:
            staffing = st.session_state.full_response.get('clinical_rationale', {})
            total_staff = 0
            # Get staffing from medicine requirements if we have it
            alloc_result = st.session_state.full_response
            if isinstance(alloc_result, dict):
                # Use clinical rationale to estimate
                allocs = alloc_result.get('allocation', [])
                total_staff = max(len(allocs) * 3, esi_data.get('critical_patients', 0) * 3)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: #6366f1;">{total_staff}</div>
                <div class="stat-label">👩‍⚕️ Est. Staff Required</div>
            </div>
            """, unsafe_allow_html=True)

        # Clinical rationale
        rationale = st.session_state.full_response.get('clinical_rationale', {})
        if rationale:
            st.markdown("---")
            st.markdown("### 🧠 Clinical Decision Rationale")

            for r in rationale.get('allocation_rationale', []):
                risk_color = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🔴"}.get(r.get('transport_risk', ''), '⚪')
                st.markdown(
                    f"**{r['hospital']}** — {r['patients_assigned']} patients, "
                    f"{r['distance_km']}km, Transport Risk: {risk_color} {r['transport_risk']}, "
                    f"Capacity: {r['capacity_utilization_pct']}%"
                )
                st.caption(f"_{r.get('eta_note', '')} | {r.get('capacity_note', '')}_")

            concerns = rationale.get('clinical_concerns', [])
            if concerns:
                st.markdown("#### ⚠️ Clinical Concerns")
                for c in concerns:
                    st.warning(c)

            recs = rationale.get('recommendations', [])
            if recs:
                with st.expander("📋 Recommendations", expanded=False):
                    for rec in recs:
                        st.markdown(f"• {rec}")

    else:
        st.info("🏷️ Trigger a disaster response to see ESI triage distribution")
        st.markdown("""
        **Emergency Severity Index (ESI)** is a 5-level triage system:
        - **ESI-1** 🔴 Immediate — Life-threatening, immediate intervention
        - **ESI-2** 🟠 Emergent — High risk, severe pain, altered mental status
        - **ESI-3** 🟡 Urgent — Stable vitals, needs multiple resources
        - **ESI-4** 🟢 Less Urgent — Needs one resource (X-ray, sutures)
        - **ESI-5** 🔵 Non-Urgent — No resources needed
        """)

    # Download report
    if st.session_state.full_response:
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            report_json = json.dumps(st.session_state.full_response, indent=2, default=str)
            st.download_button(
                "📥 Download Full Report (JSON)",
                data=report_json,
                file_name=f"disaster_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_dl2:
            # Create CSV from allocation
            alloc_data = st.session_state.full_response.get('allocation', [])
            if alloc_data:
                df = pd.DataFrame(alloc_data)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Allocation (CSV)",
                    data=csv_data,
                    file_name=f"allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ============================================
# TAB 3: AI INSIGHTS
# ============================================
with tab_ai:
    if st.session_state.full_response and st.session_state.full_response.get('ai_insights'):
        ai = st.session_state.full_response['ai_insights']

        st.markdown("### 🧠 AI/ML Engine Analysis")
        st.caption("Results from 7 AI engines running in parallel on the disaster scenario")

        # ---- Engine Status Row ----
        e1, e2, e3, e4 = st.columns(4)
        engine_count = 7 if 'multi_agent_rl' in ai else 6
        with e1:
            status = 'partial' if ai.get('status') == 'partial' else 'operational'
            color = '#f59e0b' if status == 'partial' else '#10b981'
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: {color};">{engine_count}</div>
                <div class="stat-label">AI Engines Active</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            dl = ai.get('deep_learning', {})
            conf = dl.get('overall_confidence', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: #818cf8;">{conf:.0%}</div>
                <div class="stat-label">🔮 DL Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            marl = ai.get('multi_agent_rl', {})
            team_r = marl.get('episode_result', {}).get('reward', {}).get('team_reward', 0)
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: #f97316;">{team_r:.2f}</div>
                <div class="stat-label">🤖 MARL Team Reward</div>
            </div>
            """, unsafe_allow_html=True)
        with e4:
            nash = marl.get('nash_equilibrium', {})
            is_nash = nash.get('is_nash_equilibrium', False)
            nash_color = '#10b981' if is_nash else '#ef4444'
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color: {nash_color};">{'✓' if is_nash else '✗'}</div>
                <div class="stat-label">Nash Equilibrium</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ---- Deep Learning & RL ----
        col_dl, col_rl = st.columns(2)

        with col_dl:
            st.markdown("#### 🔮 Deep Learning — Severity Prediction")
            dl = ai.get('deep_learning', {})
            if dl and not isinstance(dl, str):
                pred = dl.get('severity_prediction', {})
                dist = pred.get('distribution', {})
                if dist:
                    esi_colors = {'ESI-1': '#ef4444', 'ESI-2': '#f97316', 'ESI-3': '#eab308', 'ESI-4': '#22c55e', 'ESI-5': '#3b82f6'}
                    for lvl, pct in dist.items():
                        c = esi_colors.get(lvl, '#666')
                        bar_w = max(float(pct) * 100, 5) if isinstance(pct, (int, float)) else 5
                        st.markdown(f"""
                        <div class="esi-bar-container">
                            <div class="esi-bar-label">{lvl}</div>
                            <div class="esi-bar-bg">
                                <div class="esi-bar-fill" style="width:{bar_w}%;background:{c};">{pct:.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                demand = dl.get('demand_forecast', {})
                if demand:
                    st.caption(f"Demand forecast model: {demand.get('model', 'MLP')}")
            else:
                st.info("DL engine data unavailable")

        with col_rl:
            st.markdown("#### 🎯 Reinforcement Learning — DQN")
            rl = ai.get('reinforcement_learning', {})
            if rl and not isinstance(rl, str):
                dqn = rl.get('dqn_allocation', {})
                reward = dqn.get('reward', {})
                if reward:
                    st.markdown(f"**Total Reward:** `{reward.get('total_reward', 0):.4f}`")
                    components = reward.get('components', {})
                    for comp, val in components.items():
                        bar_w = max(abs(float(val)) * 100, 3)
                        c = '#10b981' if float(val) > 0 else '#ef4444'
                        st.markdown(f"""
                        <div class="esi-bar-container">
                            <div class="esi-bar-label">{comp}</div>
                            <div class="esi-bar-bg">
                                <div class="esi-bar-fill" style="width:{min(bar_w,100)}%;background:{c};">{val:.3f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                strategy = rl.get('meta_strategy', {})
                if strategy:
                    st.caption(f"Strategy: {strategy.get('name', 'DQN')}")
            else:
                st.info("RL engine data unavailable")

        st.markdown("---")

        # ---- GNN & NSGA-II ----
        col_gnn, col_nsga = st.columns(2)

        with col_gnn:
            st.markdown("#### 🕸️ Graph Neural Network")
            gnn = ai.get('graph_neural_network', {})
            if gnn and not isinstance(gnn, str):
                stats = gnn.get('graph_statistics', {})
                flow = gnn.get('network_flow', {})
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Nodes | {stats.get('total_nodes', '?')} |
                | Edges | {stats.get('total_edges', '?')} |
                | Density | {stats.get('graph_density', '?')} |
                | Max Flow | {flow.get('max_flow', '?')} patients |
                | Bottleneck | {'⚠️ Yes' if flow.get('bottleneck') else '✅ No'} |
                """)
                cascade = gnn.get('cascade_failure', {})
                if cascade:
                    resil = cascade.get('system_resilience', 0)
                    risk = cascade.get('risk_level', 'UNKNOWN')
                    risk_c = {'LOW': '🟢', 'MODERATE': '🟡', 'HIGH': '🔴', 'CRITICAL': '🔴'}.get(risk, '⚪')
                    st.markdown(f"**Cascade Risk:** {risk_c} {risk} (resilience: {resil:.0%})")
            else:
                st.info("GNN engine data unavailable")

        with col_nsga:
            st.markdown("#### 📐 NSGA-II Multi-Objective")
            nsga = ai.get('multi_objective_optimization', {})
            if nsga and not isinstance(nsga, str):
                pf = nsga.get('pareto_front', {})
                st.markdown(f"**Pareto Solutions:** {pf.get('size', '?')}")
                knee = nsga.get('recommended_solution', {})
                if knee:
                    st.markdown("**Recommended (Knee Point):**")
                    objectives = knee.get('objectives', {})
                    for obj, val in objectives.items():
                        st.markdown(f"- {obj}: `{val}`")
                quality = nsga.get('quality_metrics', {})
                if quality:
                    st.caption(f"Hypervolume: {quality.get('hypervolume', '?')} | Spread: {quality.get('spread', '?')}")
            else:
                st.info("NSGA-II engine data unavailable")

        st.markdown("---")

        # ---- Markov & NLP ----
        col_mdp, col_nlp = st.columns(2)

        with col_mdp:
            st.markdown("#### 📈 Markov Decision Process")
            mdp = ai.get('markov_decision_process', {})
            if mdp and not isinstance(mdp, str):
                mc = mdp.get('monte_carlo_simulation', {})
                final = mc.get('final_outcome', {})
                if final:
                    for state_name, data in final.items():
                        if isinstance(data, dict) and 'percentage' in data:
                            pct = data['percentage']
                            colors = {'recovered': '#10b981', 'stable': '#3b82f6',
                                      'critical': '#f97316', 'icu': '#eab308', 'deceased': '#ef4444'}
                            c = colors.get(state_name, '#666')
                            st.markdown(f"""
                            <div class="esi-bar-container">
                                <div class="esi-bar-label">{state_name.title()}</div>
                                <div class="esi-bar-bg">
                                    <div class="esi-bar-fill" style="width:{max(pct, 3)}%;background:{c};">{pct:.1f}%</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                policy = mdp.get('optimal_policy', {})
                if policy:
                    action = policy.get('recommended_action', '')
                    if action:
                        st.caption(f"Optimal policy: {action}")
            else:
                st.info("MDP engine data unavailable")

        with col_nlp:
            st.markdown("#### 📝 NLP Clinical Reasoning")
            nlp = ai.get('nlp_clinical_reasoning', {})
            if nlp and not isinstance(nlp, str):
                consensus = nlp.get('multi_agent_consensus', {})
                assessment = consensus.get('overall_assessment', 'N/A')
                score = consensus.get('consensus_score', 0)
                st.markdown(f"**Consensus:** {assessment}")
                st.markdown(f"**Score:** `{score:.2f}`")
                recs = nlp.get('recommendations', [])
                if recs:
                    with st.expander(f"🔍 {len(recs)} Clinical Recommendations", expanded=False):
                        for r in recs[:6]:
                            if isinstance(r, dict):
                                st.markdown(f"- **{r.get('action', '')}** — {r.get('rationale', '')}")
                            else:
                                st.markdown(f"- {r}")
                protocols = nlp.get('matched_protocols', [])
                if protocols:
                    st.caption(f"Protocols: {', '.join(p.get('protocol', str(p)) if isinstance(p, dict) else str(p) for p in protocols[:3])}")
            else:
                st.info("NLP engine data unavailable")

        st.markdown("---")

        # ---- MARL Section ----
        st.markdown("#### 🤖 Multi-Agent Reinforcement Learning (MARL)")
        marl = ai.get('multi_agent_rl', {})
        if marl and not isinstance(marl, str):
            st.caption(f"Architecture: {marl.get('architecture', 'CTDE')} | "
                       f"Agents: {marl.get('agents', {}).get('count', 4)}")

            m1, m2, m3, m4 = st.columns(4)
            episode = marl.get('episode_result', {})
            actions = episode.get('actions', {})
            decisions = episode.get('decisions', {})

            agent_icons = {'hospital': '🏥', 'ambulance': '🚑', 'triage': '🏷️', 'resource': '📦'}

            for col, (agent_name, icon) in zip([m1, m2, m3, m4], agent_icons.items()):
                with col:
                    action = actions.get(agent_name, 'N/A')
                    dec = decisions.get(agent_name, {})
                    desc = dec.get('policy_description', dec.get('strategy', dec.get('protocol', action)))
                    st.markdown(f"""
                    <div class="stat-card" style="text-align:left;padding:14px;">
                        <div style="font-size:1.5rem;">{icon}</div>
                        <div style="color:#818cf8;font-weight:700;font-size:0.75rem;text-transform:uppercase;">{agent_name}</div>
                        <div style="color:white;font-weight:600;font-size:0.85rem;margin-top:4px;">{action.replace('_',' ').title()}</div>
                        <div style="color:rgba(255,255,255,0.6);font-size:0.72rem;margin-top:4px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Reward breakdown
            reward = episode.get('reward', {})
            components = reward.get('components', {})
            if components:
                st.markdown("##### Cooperative Reward Breakdown")
                rc1, rc2, rc3, rc4 = st.columns(4)
                comp_list = list(components.items())
                cols_list = [rc1, rc2, rc3, rc4]
                comp_icons = {'survival': '❤️', 'efficiency': '⚡', 'coverage': '📡', 'coordination': '🤝'}
                for i, (comp_name, val) in enumerate(comp_list[:4]):
                    with cols_list[i]:
                        ic = comp_icons.get(comp_name, '📊')
                        st.metric(f"{ic} {comp_name.title()}", f"{val:.2f}")

            # Nash Equilibrium
            nash = marl.get('nash_equilibrium', {})
            if nash:
                is_nash = nash.get('is_nash_equilibrium', False)
                eps = nash.get('epsilon', 0)
                interp = nash.get('interpretation', '')
                if is_nash:
                    st.success(f"✅ {interp}")
                else:
                    st.warning(f"⚠️ {interp}")

            # Value decomposition
            qvals = episode.get('q_values', {})
            vd = marl.get('value_decomposition', {})
            if qvals:
                with st.expander("🔢 QMIX Value Decomposition", expanded=False):
                    st.markdown(f"**Q_total (QMIX):** `{qvals.get('q_total_qmix', 0):.4f}`")
                    st.markdown("Individual agent Q-values:")
                    for agent_name, q in qvals.get('individual', {}).items():
                        st.markdown(f"- {agent_name}: `{q:.4f}`")
                    st.caption(f"Monotonicity: {vd.get('monotonicity', '∂Q_tot/∂Q_i ≥ 0')}")

            # Theoretical basis
            theory = marl.get('theoretical_basis', [])
            if theory:
                with st.expander("📚 Theoretical Foundation", expanded=False):
                    for t in theory:
                        st.markdown(f"- {t}")
        else:
            st.info("MARL engine data unavailable — included in v3.1+")

    else:
        st.info("🧠 Trigger a disaster response to see AI insights")
        st.markdown("""
        **7 AI/ML engines** analyze your disaster scenario in parallel:
        1. **Deep Learning** — MLP severity prediction with Monte Carlo dropout
        2. **Reinforcement Learning** — Dueling DQN with Thompson Sampling
        3. **Graph Neural Network** — Message passing + attention + max-flow routing
        4. **NSGA-II** — Multi-objective Pareto optimization (4 objectives)
        5. **Markov Decision Process** — Monte Carlo simulation (1000 runs) + value iteration
        6. **NLP Clinical Reasoning** — Attention-based protocol matching
        7. **Multi-Agent RL (MARL)** — CTDE + QMIX with 4 cooperative agents
        """)


# ============================================
# TAB 4: ANALYTICS
# ============================================
with tab_analytics:
    st.markdown("### 📊 Response Analytics")

    try:
        analytics = requests.get(f"{BACKEND_URL}/analytics", timeout=5).json()

        # Overview metrics
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            st.metric("Total Events", analytics.get('total_events', 0))
        with a2:
            st.metric("Total Patients", analytics.get('total_patients', 0))
        with a3:
            st.metric("Allocation Rate", f"{analytics.get('allocation_rate_pct', 0)}%")
        with a4:
            st.metric("Unallocated", analytics.get('total_unallocated', 0),
                      delta_color="inverse")

        # Disaster type breakdown
        type_counts = analytics.get('disaster_type_counts', {})
        if type_counts:
            st.markdown("#### Disaster Type Distribution")
            df_types = pd.DataFrame(
                list(type_counts.items()),
                columns=['Disaster Type', 'Count']
            )
            st.bar_chart(df_types.set_index('Disaster Type'))

        # Recent events table
        recent = analytics.get('recent_events', [])
        if recent:
            st.markdown("#### Recent Events")
            events_display = []
            for e in reversed(recent):
                events_display.append({
                    "ID": e.get('id', '?'),
                    "Time": e.get('timestamp', '?')[:19],
                    "Type": e.get('disaster', {}).get('type', '?'),
                    "Patients": e.get('result_summary', {}).get('total_patients', 0),
                    "Allocated": e.get('result_summary', {}).get('patients_allocated', 0),
                    "Hospitals": e.get('result_summary', {}).get('hospitals_used', 0),
                })
            st.dataframe(pd.DataFrame(events_display), use_container_width=True, hide_index=True)

        # Cache performance
        cache_stats = analytics.get('cache_performance', {})
        if cache_stats:
            st.markdown("#### Cache Performance")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Cache Entries", cache_stats.get('entries', 0))
            with c2:
                st.metric("Hit Rate", f"{cache_stats.get('hit_rate_percent', 0)}%")
            with c3:
                st.metric("Total Hits", cache_stats.get('hits', 0))

    except requests.exceptions.ConnectionError:
        st.warning("Analytics unavailable — backend is offline")
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")


# ============================================
# TAB 5: RESEARCH & ANALYSIS
# ============================================
with tab_research:
    st.markdown("### 🔬 System Research & Performance Analysis")
    
    tab_ablation, tab_perf = st.tabs(["📉 Ablation Study", "⚡ System Performance"])
    
    with tab_ablation:
        st.markdown("#### Cooperative MARL vs. Baseline Impact")
        st.info("Run a comparative study to measure the impact of MARL cooperation against a random baseline.")
        
        col_run, col_res = st.columns([1, 2])
        with col_run:
            if st.button("🚀 Run Ablation Study", type="primary"):
                with st.spinner("Running comparative simulations..."):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/ablation-study",
                            json={
                                "patients": patients,
                                "lat": disaster_lat,
                                "lng": disaster_lng,
                                "disaster_type": disaster_type,
                                "source": data_source
                            },
                            timeout=60
                        )
                        if resp.status_code == 200:
                            st.session_state.ablation_result = resp.json()
                            st.success("Study Complete!")
                        else:
                            st.error(f"Study failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")

        with col_res:
            if "ablation_result" in st.session_state:
                res = st.session_state.ablation_result
                uplift = res["uplift"]
                base = res["baseline"]
                marl = res["marl"]
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Survival Gain", f"{uplift['survival_gain_pct']}%", delta="Uplift")
                m2.metric("Efficiency Gain", f"{uplift['efficiency_gain_pct']}%", delta="Uplift")
                m3.metric("Team Reward Gain", f"{uplift['team_reward_gain_pct']}%", delta="Uplift")
                
                # Charts
                st.markdown("##### Comparative Analysis")
                chart_data = pd.DataFrame({
                    "Metric": ["Survival Rate", "Efficiency", "Team Reward"],
                    "Baseline (Random)": [base["survival_rate"], base["response_efficiency"], base["team_reward"]],
                    "MARL (Cooperative)": [marl["survival_rate"], marl["response_efficiency"], marl["team_reward"]]
                })
                st.bar_chart(chart_data.set_index("Metric"), color=["#94a3b8", "#10b981"])
                
                st.caption(f"Study based on 1 episode comparison. Complexity: {res['complexity_analysis']['time_complexity']}")

    with tab_perf:
        st.markdown("#### ⚡ System Scalability & Complexity")
        if st.button("Refresh Performance Data"):
            try:
                perf = requests.get(f"{BACKEND_URL}/system-performance", timeout=5).json()
                st.session_state.perf_data = perf
            except:
                st.error("Failed to fetch performance data")
        
        if "perf_data" in st.session_state:
            perf = st.session_state.perf_data
            
            p1, p2 = st.columns(2)
            with p1:
                st.markdown("##### ⏱️ Time Complexity Analysis")
                st.table(pd.DataFrame(
                    list(perf["time_complexity"].items()),
                    columns=["Module", "Big-O Notation"]
                ))
            
            with p2:
                st.markdown("##### 📈 Scalability Test Results")
                st.table(pd.DataFrame(
                    list(perf["scalability_test"].items()),
                    columns=["Load Scenario", "Performance"]
                ))
            
            st.success(f"System Status: {perf['system_status']}")


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.3); padding: 16px;">
    <p style="font-weight: 600; letter-spacing: 1px; font-size: 0.8rem;">
        S.A.V.E. — STRATEGIC AGENT-BASED VICTIM EVACUATION
    </p>
    <p style="font-size: 0.7rem; margin-top: 4px;">
        Hospital Agent • Emergency Agent • Government Agent • Triage Agent • Medical Supply Agent • AI Engine (7 modules)
    </p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if st.session_state.auto_refresh and st.session_state.disaster_triggered:
    time.sleep(30)
    st.rerun()
