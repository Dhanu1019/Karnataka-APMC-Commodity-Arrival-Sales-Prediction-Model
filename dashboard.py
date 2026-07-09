import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
 
st.set_page_config(
    page_title="Karnataka APMC Sales Predictor",
    page_icon="🌾",
    layout="wide"
)
 
# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")
 
@st.cache_resource
def load_model():
    return joblib.load("outputs/best_model.pkl")
 
if not os.path.exists("outputs/best_model.pkl"):
    st.error("⚠️ Model not found. Please run steps 1–3 first:\n"
             "```\npython step1_preprocess.py\n"
             "python step2_eda.py\n"
             "python step3_train.py\n```")
    st.stop()
 
df    = load_data()
model = load_model()
 
month_map = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"
}
rev_month = {v: k for k, v in month_map.items()}
 
# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌾 Karnataka APMC Commodity Arrival Predictor")
st.caption("Sales Prediction Model — Task 03 | Data: Karnataka Dept. of Agriculture 2012")
st.divider()
 
# ── Sidebar: Model Performance ────────────────────────────────────────────────
st.sidebar.header("📊 Model Performance")
if os.path.exists("outputs/model_comparison.csv"):
    metrics = pd.read_csv("outputs/model_comparison.csv")
    st.sidebar.dataframe(metrics.set_index("Model").round(3), use_container_width=True)
 
st.sidebar.divider()
st.sidebar.header("📁 Dataset Stats")
st.sidebar.metric("Total Records",    f"{len(df):,}")
st.sidebar.metric("Commodities",      df["Commodity"].nunique())
st.sidebar.metric("Districts",        df["District"].nunique())
st.sidebar.metric("Markets",          df["Market"].nunique())
 
# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Live Prediction", "📈 EDA Charts", "📋 Raw Data"])
 
# ════════════════════════════════════════════════
# TAB 1 : LIVE PREDICTION
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Commodity Arrival at any Karnataka APMC Market")
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        district = st.selectbox("District", sorted(df["District"].unique()))
        filtered_taluks = sorted(df[df["District"] == district]["Taluk"].unique())
        taluk = st.selectbox("Taluk", filtered_taluks)
 
    with col2:
        filtered_markets = sorted(
            df[(df["District"] == district) & (df["Taluk"] == taluk)]["Market"].unique()
        )
        market = st.selectbox("Market", filtered_markets if filtered_markets else df["Market"].unique())
        commodity = st.selectbox("Commodity", sorted(df["Commodity"].unique()))
 
    with col3:
        unit      = st.selectbox("Unit", ["Quintal", "Numbers", "Thousands"])
        month_sel = st.selectbox("Month", list(month_map.values()))
        month_num = rev_month[month_sel]
 
    if st.button("🌾 Predict Arrival", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "District" : district,
            "Taluk"    : taluk,
            "Market"   : market,
            "Commodity": commodity,
            "Unit"     : unit,
            "Month_Num": month_num,
        }])
        log_pred  = model.predict(row)[0]
        pred_val  = np.expm1(log_pred)
 
        st.success(f"### 📦 Predicted Arrival: **{pred_val:,.0f} {unit}**")
 
        # Show historical average for comparison
        hist = df[
            (df["District"]  == district) &
            (df["Commodity"] == commodity)
        ]["Arrival"]
        if len(hist) > 0:
            st.info(f"📊 Historical avg for {commodity} in {district}: "
                    f"**{hist.mean():,.0f} {unit}** "
                    f"(based on {len(hist)} records)")
 
    st.divider()
 
    # Mini trend chart for selected commodity × district
    st.subheader(f"📈 Monthly Trend — {commodity} in {district}")
    trend = (
        df[(df["District"] == district) & (df["Commodity"] == commodity)]
        .groupby("Month_Num")["Arrival"].mean()
        .reindex(range(1, 13))
    )
    if trend.notna().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(trend.index, trend.values, marker="o",
                color="#2563eb", linewidth=2.5, markersize=7)
        ax.fill_between(trend.index, trend.values, alpha=0.15, color="#2563eb")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(list(month_map.values()))
        ax.set_ylabel(f"Avg Arrival ({unit})")
        ax.set_title(f"{commodity} — {district} — Monthly Average Arrival")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("No historical data for this District + Commodity combination.")
 
# ════════════════════════════════════════════════
# TAB 2 : EDA CHARTS
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Exploratory Data Analysis")
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        if os.path.exists("outputs/eda_top_commodities.png"):
            st.image("outputs/eda_top_commodities.png", use_container_width=True)
        if os.path.exists("outputs/eda_top_districts.png"):
            st.image("outputs/eda_top_districts.png", use_container_width=True)
 
    with col_b:
        if os.path.exists("outputs/eda_monthly_trend.png"):
            st.image("outputs/eda_monthly_trend.png", use_container_width=True)
        if os.path.exists("outputs/eda_distribution.png"):
            st.image("outputs/eda_distribution.png", use_container_width=True)
 
    st.divider()
    st.subheader("Model Results")
 
    col_c, col_d = st.columns(2)
    with col_c:
        if os.path.exists("outputs/model_comparison.png"):
            st.image("outputs/model_comparison.png", use_container_width=True)
    with col_d:
        if os.path.exists("outputs/actual_vs_predicted.png"):
            st.image("outputs/actual_vs_predicted.png", use_container_width=True)
 
    if os.path.exists("outputs/feature_importance.png"):
        st.image("outputs/feature_importance.png", use_container_width=True)
 
# ════════════════════════════════════════════════
# TAB 3 : RAW DATA
# ════════════════════════════════════════════════
with tab3:
    st.subheader("Browse Cleaned Dataset")
 
    f_district  = st.multiselect("Filter by District",  sorted(df["District"].unique()))
    f_commodity = st.multiselect("Filter by Commodity", sorted(df["Commodity"].unique()))
 
    view = df.copy()
    if f_district:
        view = view[view["District"].isin(f_district)]
    if f_commodity:
        view = view[view["Commodity"].isin(f_commodity)]
 
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=400)
    st.caption(f"Showing {len(view):,} of {len(df):,} records")
 
st.divider()
st.caption("Built for Task-03 · Karnataka Dept. of Agriculture & Farmers Welfare · 2012 APMC Data")