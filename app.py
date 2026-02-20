"""
Mining Quality Control Dashboard — Streamlit App
=================================================
Deploy on Streamlit Cloud after pushing to GitHub.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import json
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mining Quality Dashboard",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f2f5; }
    .block-container { padding-top: 1.5rem; }

    .status-green  { background:#d5f5e3; border:2px solid #27ae60;
                     border-radius:12px; padding:16px; text-align:center; }
    .status-amber  { background:#fef9e7; border:2px solid #f39c12;
                     border-radius:12px; padding:16px; text-align:center; }
    .status-red    { background:#fadbd8; border:2px solid #e74c3c;
                     border-radius:12px; padding:16px; text-align:center; }

    .kpi-card { background:white; border-radius:12px; padding:18px;
                box-shadow:0 2px 10px rgba(0,0,0,0.08); text-align:center; }
    .kpi-value { font-size:2.2rem; font-weight:700; }
    .kpi-label { font-size:0.8rem; color:#95a5a6; margin-top:4px; }

    .section-title { font-size:1rem; font-weight:700; color:#2c3e50;
                     margin-bottom:8px; }
    .alert-badge-green { background:#27ae60; color:white; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600; }
    .alert-badge-amber { background:#f39c12; color:white; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600; }
    .alert-badge-red   { background:#e74c3c; color:white; padding:4px 14px;
                         border-radius:20px; font-size:0.8rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load saved model and scaler from disk."""
    model  = joblib.load("model/xgb_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/config.json") as f:
        config = json.load(f)
    return model, scaler, config

# ── Load Historical Data ──────────────────────────────────────────────────────
@st.cache_data
def load_history():
    """Load scored history CSV for trend charts."""
    if os.path.exists("data/scored_history.csv"):
        df = pd.read_csv("data/scored_history.csv", parse_dates=["timestamp"])
        return df
    return pd.DataFrame()


# ── Scoring Function ──────────────────────────────────────────────────────────
def score_reading(model, scaler, features, values: dict, threshold: float):
    """Score a single sensor reading. Returns prob, alert level, and status."""
    row        = pd.DataFrame([{f: values.get(f, 0.0) for f in features}])
    row_scaled = scaler.transform(row)
    row_df     = pd.DataFrame(row_scaled, columns=features)
    prob       = float(model.predict_proba(row_df)[0, 1])

    if prob >= threshold:
        alert, icon, css = "RED",   "🚨", "status-red"
        msg = "INTERVENTION REQUIRED"
    elif prob >= threshold * 0.7:
        alert, icon, css = "AMBER", "⚠️", "status-amber"
        msg = "MONITOR CLOSELY"
    else:
        alert, icon, css = "GREEN", "✅", "status-green"
        msg = "NORMAL OPERATION"

    # Feature importance × deviation → top drivers
    importance = model.feature_importances_
    deviation  = np.abs(row_scaled[0])
    risk_score = importance * deviation
    top_idx    = np.argsort(risk_score)[::-1][:3]
    drivers    = [(features[i], float(risk_score[i])) for i in top_idx]

    return prob, alert, icon, msg, css, drivers


# ── Action Map ────────────────────────────────────────────────────────────────
ACTION_MAP = {
    "Starch Flow"      : "Increase starch dosing to suppress silica attachment",
    "Amina Flow"       : "Reduce amina flow — currently driving silica up",
    "Ore Pulp pH"      : "Adjust pH to optimal flotation range (9.5–10.5)",
    "Ore Pulp Density" : "Check feed density — dilute if above optimal",
    "Ore Pulp Flow"    : "Reduce feed flow rate to improve residence time",
    "Avg_Air_Flow"     : "Increase air flow to improve bubble formation",
    "Air_Flow_Std"     : "Stabilise air flow across columns — check valves",
    "Level_Std"        : "Equalise column levels — check level controllers",
}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load Assets ───────────────────────────────────────────────────────────
    try:
        model, scaler, config = load_model()
        features  = config["features"]
        threshold = config["deployment_threshold"]
        model_ok  = True
    except Exception as e:
        st.error(f"⚠️ Could not load model: {e}\n\nRun `save_model.py` first.")
        model_ok = False
        return

    history = load_history()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/mine-cart.png", width=60)
        st.title("⛏️ Mining QC")
        st.markdown("**Flotation Plant Monitor**")
        st.divider()

        st.markdown("### ⚙️ Model Info")
        st.markdown(f"**Model:** XGBoost")
        st.markdown(f"**Threshold:** `{threshold:.2f}`")
        st.markdown(f"**Features:** {len(features)}")
        st.divider()

        st.markdown("### 📋 Alert Thresholds")
        st.markdown(f"🔴 **RED**   ≥ `{threshold:.2f}`")
        st.markdown(f"🟡 **AMBER** ≥ `{threshold*0.7:.2f}`")
        st.markdown(f"🟢 **GREEN** < `{threshold*0.7:.2f}`")
        st.divider()

        page = st.radio("Navigation",
                        ["🏠 Live Scoring", "📈 Historical Trends",
                         "🔍 Feature Inspector", "📊 Drift Monitor"])

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1: LIVE SCORING
    # ══════════════════════════════════════════════════════════════════════════
    if page == "🏠 Live Scoring":
        st.title("⛏️ Mining Quality Control — Live Scoring")
        st.markdown("Enter current sensor readings to get an instant quality prediction.")
        st.divider()

        # ── Input Section ─────────────────────────────────────────────────────
        st.markdown("### 📡 Enter Sensor Readings")

        with st.form("sensor_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Feed & Reagents**")
                iron_feed    = st.number_input("% Iron Feed",      2.0, 70.0, 55.0, 0.1)
                silica_feed  = st.number_input("% Silica Feed",    1.0, 30.0, 12.0, 0.1)
                starch_flow  = st.number_input("Starch Flow",      0.0, 5000.0, 2500.0, 10.0)
                amina_flow   = st.number_input("Amina Flow",       0.0, 800.0, 300.0, 5.0)

            with col2:
                st.markdown("**Pulp Properties**")
                pulp_flow    = st.number_input("Ore Pulp Flow",    200.0, 600.0, 400.0, 1.0)
                pulp_ph      = st.number_input("Ore Pulp pH",      8.0, 11.0, 9.8, 0.1)
                pulp_density = st.number_input("Ore Pulp Density", 1.5, 2.5, 1.75, 0.01)

            with col3:
                st.markdown("**Column Parameters**")
                avg_air_flow   = st.number_input("Avg Air Flow",   200.0, 400.0, 290.0, 1.0)
                avg_level      = st.number_input("Avg Column Level", 400.0, 700.0, 540.0, 1.0)
                air_flow_std   = st.number_input("Air Flow Std",   0.0, 50.0, 10.0, 0.5)
                level_std      = st.number_input("Level Std",      0.0, 80.0, 20.0, 0.5)

            submitted = st.form_submit_button("🔍 Predict Quality", use_container_width=True,
                                              type="primary")

        if submitted:
            # Build reading dict — map to feature names
            reading = {
                "% Iron Feed"      : iron_feed,
                "% Silica Feed"    : silica_feed,
                "Starch Flow"      : starch_flow,
                "Amina Flow"       : amina_flow,
                "Ore Pulp Flow"    : pulp_flow,
                "Ore Pulp pH"      : pulp_ph,
                "Ore Pulp Density" : pulp_density,
                "Avg_Air_Flow"     : avg_air_flow,
                "Avg_Column_Level" : avg_level,
                "Air_Flow_Std"     : air_flow_std,
                "Level_Std"        : level_std,
                # Derived
                "Total_Reagent_Flow"  : starch_flow + amina_flow,
                "Iron_Recovery_Rate"  : 100.0,  # placeholder
                "Silica_Rejection_Rate": 80.0,  # placeholder
                "Selectivity_Index"   : 30.0,   # placeholder
            }

            prob, alert, icon, msg, css, drivers = score_reading(
                model, scaler, features, reading, threshold)

            st.divider()
            st.markdown("### 🎯 Prediction Result")

            # KPI Row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-value" style="color:{'#e74c3c' if alert=='RED' else '#f39c12' if alert=='AMBER' else '#27ae60'}">
                    {prob*100:.1f}%
                  </div>
                  <div class="kpi-label">Failure Probability</div>
                </div>""", unsafe_allow_html=True)

            with k2:
                st.markdown(f"""
                <div class="{css}">
                  <div style="font-size:2rem">{icon}</div>
                  <div style="font-weight:700;font-size:1rem;margin-top:6px">{alert}</div>
                  <div style="font-size:0.8rem;color:#555">{msg}</div>
                </div>""", unsafe_allow_html=True)

            with k3:
                gauge = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = prob * 100,
                    number= {"suffix": "%", "font": {"size": 24}},
                    gauge = {
                        "axis"  : {"range": [0, 100]},
                        "bar"   : {"color": "#e74c3c" if alert=="RED" else "#f39c12" if alert=="AMBER" else "#27ae60"},
                        "steps" : [
                            {"range": [0, threshold*70],    "color": "#d5f5e3"},
                            {"range": [threshold*70, threshold*100], "color": "#fef9e7"},
                            {"range": [threshold*100, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 3},
                                      "value": threshold * 100}
                    }
                ))
                gauge.update_layout(height=160, margin=dict(t=10, b=10, l=20, r=20))
                st.plotly_chart(gauge, use_container_width=True)

            with k4:
                st.markdown("**💡 Recommended Actions**")
                for feat, _ in drivers:
                    for key, action in ACTION_MAP.items():
                        if key.lower() in feat.lower():
                            st.markdown(f"• {action}")
                            break

            # Risk Drivers
            st.markdown("### 🔍 Top Risk Drivers")
            driver_df = pd.DataFrame(drivers, columns=["Feature", "Risk Score"])
            fig_bar = px.bar(driver_df, x="Risk Score", y="Feature",
                             orientation="h", color="Risk Score",
                             color_continuous_scale="RdYlGn_r",
                             title="Feature Risk Contribution")
            fig_bar.update_layout(height=250, showlegend=False,
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 2: HISTORICAL TRENDS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📈 Historical Trends":
        st.title("📈 Historical Quality Trends")

        if history.empty:
            st.warning("No historical data found. Ensure `data/scored_history.csv` exists.")
            return

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            hours = st.slider("Show last N hours", 24, 720, 168)
        with col2:
            show_threshold = st.checkbox("Show thresholds", value=True)

        plot_data = history.tail(hours).copy()

        # Main trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_data["timestamp"], y=plot_data["failure_probability"],
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2),
            marker=dict(
                color=plot_data["alert_level"].map(
                    {"GREEN": "#27ae60", "AMBER": "#f39c12", "RED": "#e74c3c"}),
                size=6),
            name="Failure Probability"
        ))
        if show_threshold:
            fig.add_hline(y=threshold, line_dash="dash",
                          line_color="red", annotation_text="RED threshold")
            fig.add_hline(y=threshold*0.7, line_dash="dash",
                          line_color="orange", annotation_text="AMBER threshold")
        fig.update_layout(
            title=f"Failure Probability — Last {hours} Hours",
            yaxis_title="Failure Probability",
            yaxis=dict(range=[0, 1]),
            height=380,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Avg Risk", f"{plot_data['failure_probability'].mean()*100:.1f}%")
        with k2:
            st.metric("Peak Risk", f"{plot_data['failure_probability'].max()*100:.1f}%")
        with k3:
            red_count = (plot_data["alert_level"] == "RED").sum()
            st.metric("🔴 RED Alerts", int(red_count))
        with k4:
            amber_count = (plot_data["alert_level"] == "AMBER").sum()
            st.metric("🟡 AMBER Alerts", int(amber_count))

        # Alert distribution pie
        col1, col2 = st.columns(2)
        with col1:
            alert_counts = plot_data["alert_level"].value_counts()
            fig_pie = px.pie(
                values=alert_counts.values,
                names=alert_counts.index,
                color=alert_counts.index,
                color_discrete_map={"GREEN":"#27ae60","AMBER":"#f39c12","RED":"#e74c3c"},
                title="Alert Level Distribution"
            )
            fig_pie.update_layout(height=320)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Hourly heatmap
            if "timestamp" in plot_data.columns:
                plot_data["hour"] = plot_data["timestamp"].dt.hour
                hourly = plot_data.groupby("hour")["failure_probability"].mean().reset_index()
                fig_hour = px.bar(hourly, x="hour", y="failure_probability",
                                  color="failure_probability",
                                  color_continuous_scale="RdYlGn_r",
                                  title="Average Risk by Hour of Day",
                                  labels={"failure_probability": "Avg Risk"})
                fig_hour.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig_hour, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 3: FEATURE INSPECTOR
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "🔍 Feature Inspector":
        st.title("🔍 Feature Inspector")
        st.markdown("Explore how individual sensor values affect the failure prediction.")

        if history.empty:
            st.warning("No historical data found.")
            return

        feature_select = st.selectbox("Select Feature", options=features)

        if feature_select in history.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig_scatter = px.scatter(
                    history.tail(500), x=feature_select,
                    y="failure_probability",
                    color="alert_level",
                    color_discrete_map={"GREEN":"#27ae60","AMBER":"#f39c12","RED":"#e74c3c"},
                    title=f"{feature_select} vs Failure Probability",
                    trendline="ols"
                )
                fig_scatter.update_layout(height=350)
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col2:
                fig_hist = px.histogram(
                    history.tail(500), x=feature_select,
                    color="alert_level",
                    color_discrete_map={"GREEN":"#27ae60","AMBER":"#f39c12","RED":"#e74c3c"},
                    title=f"{feature_select} Distribution by Alert Level",
                    barmode="overlay", opacity=0.7
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info(f"'{feature_select}' not found in historical data.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 4: DRIFT MONITOR
    # ══════════════════════════════════════════════════════════════════════════
    elif page == "📊 Drift Monitor":
        st.title("📊 Model Drift Monitor")
        st.markdown("Monitor whether feature distributions have shifted since training.")

        if history.empty:
            st.warning("No historical data found.")
            return

        def psi(expected, actual, bins=10):
            exp_p, edges = np.histogram(expected, bins=bins, density=False)
            act_p, _     = np.histogram(actual, bins=edges, density=False)
            exp_p = np.where(exp_p == 0, 0.0001, exp_p / len(expected))
            act_p = np.where(act_p == 0, 0.0001, act_p / len(actual))
            return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))

        numeric_feats = [f for f in features if f in history.columns]
        n = len(history)
        baseline = history.iloc[:n//2]
        recent   = history.iloc[n//2:]

        psi_results = []
        for feat in numeric_feats[:12]:
            p = psi(baseline[feat].dropna().values,
                    recent[feat].dropna().values)
            status = "✅ Stable" if p < 0.10 else ("⚠️ Monitor" if p < 0.25 else "🔴 RETRAIN")
            psi_results.append({"Feature": feat, "PSI": round(p, 4), "Status": status})

        psi_df = pd.DataFrame(psi_results).sort_values("PSI", ascending=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig_psi = px.bar(
                psi_df, x="PSI", y="Feature", orientation="h",
                color="PSI", color_continuous_scale="RdYlGn_r",
                title="Population Stability Index (PSI) by Feature",
            )
            fig_psi.add_vline(x=0.10, line_dash="dash", line_color="orange",
                              annotation_text="Monitor (0.10)")
            fig_psi.add_vline(x=0.25, line_dash="dash", line_color="red",
                              annotation_text="Retrain (0.25)")
            fig_psi.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig_psi, use_container_width=True)

        with col2:
            st.markdown("### PSI Summary")
            st.dataframe(psi_df[["Feature", "PSI", "Status"]], use_container_width=True)

            max_psi = psi_df["PSI"].max()
            if max_psi > 0.25:
                st.error("🔴 **RETRAINING RECOMMENDED**\nOne or more features have drifted significantly.")
            elif max_psi > 0.10:
                st.warning("⚠️ **MONITOR CLOSELY**\nSome feature drift detected.")
            else:
                st.success("✅ **MODEL STABLE**\nNo significant drift detected.")


if __name__ == "__main__":
    main()
