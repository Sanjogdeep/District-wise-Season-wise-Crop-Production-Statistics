import math
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats

# ──────────────────────────────────────────────────────────────────────────
# Page setup & theme
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="District-wise Crop Yield Study", page_icon="🌾", layout="wide")

GOLD, LEAF, RUST, PAPER, DIM = "#D3A32E", "#7C9A5B", "#B85C3B", "#EFE8D6", "#C9BFA6"
LINE, SOIL2, SLATE = "#3A3226", "#262016", "#5C7A8A"
SEQ = [GOLD, LEAF, SLATE, RUST, "#8C7228", "#4E6339", "#6b5f45", "#9C8A6A"]

st.markdown(
    f"""
    <style>
    .stApp {{ background-color:#1D1810; color:{PAPER}; }}
    h1,h2,h3 {{ color:{PAPER} !important; font-family: Georgia, serif; }}
    [data-testid="stMetricValue"] {{ color:{GOLD}; }}
    [data-testid="stMetricLabel"] {{ color:{DIM}; }}
    [data-testid="stSidebar"] {{ background-color:{SOIL2}; border-right:1px solid {LINE}; }}
    .stTabs [data-baseweb="tab"] {{ color:{DIM}; font-size:1rem; }}
    .stTabs [aria-selected="true"] {{ color:{GOLD} !important; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:{LINE} !important; }}
    hr {{ border-color:{LINE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=DIM, family="IBM Plex Sans, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(font=dict(color=DIM)),
)

# ──────────────────────────────────────────────────────────────────────────
# Data loading — xlsx is slow (~20s+), so cache a fast on-disk copy
# ──────────────────────────────────────────────────────────────────────────
XLSX_PATH = "crop_enriched.xlsx"
PKL_CACHE_PATH = ".crop_enriched_cache.pkl"


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(PKL_CACHE_PATH) and os.path.getmtime(PKL_CACHE_PATH) >= os.path.getmtime(XLSX_PATH):
        return pd.read_pickle(PKL_CACHE_PATH)
    df = pd.read_excel(XLSX_PATH)
    try:
        df.to_pickle(PKL_CACHE_PATH)
    except Exception:
        pass
    return df


DATA_AVAILABLE = os.path.exists(XLSX_PATH)

st.title("District-wise, Season-wise Crop Production Statistics")
st.caption(
    "Sanjogdeep Singh — School of AI and Emerging Technology, LPU · "
    "Interactive field report on Indian crop-yield drivers, 1997–2015."
)

if not DATA_AVAILABLE:
    st.error(
        f"`{XLSX_PATH}` not found next to app.py — the dashboard needs it to compute live charts. "
        "Place the file in this folder and reload the page."
    )
    st.stop()

with st.spinner("Loading dataset (first run only — builds a fast cache for next time)..."):
    t0 = time.time()
    df_raw = load_data()
    load_secs = time.time() - t0

df_raw["Yield_capped"] = df_raw["Yield"].clip(upper=df_raw["Yield"].quantile(0.99))

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — global filters (slicers)
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 Filters")
    st.caption("Applied to every chart on the Dashboard tab.")

    yr_min, yr_max = int(df_raw["Crop_Year"].min()), int(df_raw["Crop_Year"].max())
    year_range = st.slider("Crop year range", yr_min, yr_max, (yr_min, yr_max))

    seasons = st.multiselect("Season", sorted(df_raw["Season"].unique()), default=[])
    regions = st.multiselect("Region", sorted(df_raw["Region"].unique()), default=[])
    categories = st.multiselect("Crop category", sorted(df_raw["Crop_Category"].unique()), default=[])
    states = st.multiselect("State", sorted(df_raw["State_Name"].unique()), default=[])

    st.caption("Leave a filter empty to include everything in that field.")
    if st.button("Reset all filters"):
        st.rerun()

mask = (df_raw["Crop_Year"].between(*year_range))
if seasons:
    mask &= df_raw["Season"].isin(seasons)
if regions:
    mask &= df_raw["Region"].isin(regions)
if categories:
    mask &= df_raw["Crop_Category"].isin(categories)
if states:
    mask &= df_raw["State_Name"].isin(states)

df = df_raw[mask]

if df.empty:
    st.warning("No records match the current filters — widen a selection in the sidebar.")
    st.stop()

# Sample for expensive per-point charts / stats so the UI stays responsive
SAMPLE_CAP = 20000
df_sample = df.sample(min(len(df), SAMPLE_CAP), random_state=42) if len(df) > SAMPLE_CAP else df

tab_dashboard, tab_predict = st.tabs(["📊 Dashboard", "🌱 Predict a Yield"])

# ──────────────────────────────────────────────────────────────────────────
# DASHBOARD TAB — everything computed live from the filtered data
# ──────────────────────────────────────────────────────────────────────────
with tab_dashboard:

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("States", df["State_Name"].nunique())
    c3.metric("Crops", df["Crop"].nunique())
    c4.metric("Years", f"{year_range[0]}–{year_range[1]}")
    c5.metric("High-yield share", f"{(df['Yield_Class'] == 1).mean() * 100:.1f}%")

    st.markdown("### Yield distribution")
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.metric("Median yield (t/ha)", f"{df['Yield'].median():.2f}")
        st.metric("Mean yield (t/ha)", f"{df['Yield'].mean():.2f}")
        skew_cols = ["Area", "Production", "Yield", "Rainfall_mm", "Soil_Quality", "Avg_Temp_C"]
        skew_vals = [df[c].skew() for c in skew_cols]
        fig = go.Figure(go.Bar(x=skew_cols, y=skew_vals, marker_color=SEQ[: len(skew_cols)]))
        fig.update_layout(**PLOTLY_LAYOUT, height=240, title="Skewness by feature")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.histogram(
            df, x="Yield_capped", nbins=60, color_discrete_sequence=[GOLD],
            labels={"Yield_capped": "Yield (t/ha, capped at 99th pct)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title="Yield histogram (capped for readability)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Capped at the 99th percentile for display only — a handful of extreme crops (e.g. sugarcane) otherwise dwarf the rest of the distribution.")

    st.markdown("### Season, region & crop-category mix")
    col1, col2, col3 = st.columns(3)
    with col1:
        vc = df["Season"].value_counts(normalize=True) * 100
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.55, marker=dict(colors=SEQ)))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Season")
        st.plotly_chart(fig, width="stretch")
    with col2:
        vc = df["Region"].value_counts(normalize=True) * 100
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.55, marker=dict(colors=SEQ)))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Region")
        st.plotly_chart(fig, width="stretch")
    with col3:
        vc = df["Crop_Category"].value_counts(normalize=True).sort_values() * 100
        fig = go.Figure(go.Bar(x=vc.values, y=vc.index, orientation="h", marker_color=LEAF))
        fig.update_layout(**PLOTLY_LAYOUT, height=280, title="Crop category (%)")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Yield trend over time")
    yearly = df.groupby("Crop_Year").agg(
        Median_Yield=("Yield_capped", "median"),
        Avg_Fertilizer=("Fertilizer_kg_ha", "mean"),
        Avg_Irrigation=("Irrigation_Pct", "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["Crop_Year"], y=yearly["Median_Yield"], name="Median yield (t/ha)", line=dict(color=GOLD), yaxis="y1"))
    fig.add_trace(go.Scatter(x=yearly["Crop_Year"], y=yearly["Avg_Fertilizer"], name="Avg fertilizer (kg/ha)", line=dict(color=LEAF), yaxis="y2"))
    fig.update_layout(
        **PLOTLY_LAYOUT, height=340,
        yaxis=dict(title="t/ha yield", gridcolor=LINE),
        yaxis2=dict(title="kg/ha fertilizer", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Computed live from your filtered rows, grouped by Crop_Year — updates with every slicer change above.")

    st.markdown("### State & crop leaderboards")
    col1, col2 = st.columns(2)
    MIN_N = 50
    with col1:
        state_stats = df.groupby("State_Name").agg(Median_Yield=("Yield_capped", "median"), N=("Yield", "size"))
        state_stats = state_stats[state_stats["N"] >= MIN_N].sort_values("Median_Yield")
        top_bottom = pd.concat([state_stats.head(8), state_stats.tail(8)]).sort_values("Median_Yield")
        fig = go.Figure(go.Bar(x=top_bottom["Median_Yield"], y=top_bottom.index, orientation="h", marker_color=GOLD))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title=f"Lowest & highest median-yield states (min {MIN_N} records)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        crop_stats = df.groupby("Crop").agg(Median_Yield=("Yield_capped", "median"), N=("Yield", "size"))
        crop_stats = crop_stats[crop_stats["N"] >= MIN_N].sort_values("Median_Yield", ascending=False).head(12).sort_values("Median_Yield")
        fig = go.Figure(go.Bar(x=crop_stats["Median_Yield"], y=crop_stats.index, orientation="h", marker_color=LEAF))
        fig.update_layout(**PLOTLY_LAYOUT, height=420, title=f"Top 12 crops by median yield (min {MIN_N} records)")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Correlation & relationships")
    col1, col2 = st.columns([1, 1.2])
    with col1:
        num_cols = ["Yield", "Area", "Production", "Rainfall_mm", "Soil_Quality", "Fertilizer_kg_ha", "Irrigation_Pct", "Avg_Temp_C"]
        corr = df_sample[num_cols].corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title="Pearson correlation matrix")
        st.plotly_chart(fig, width="stretch")
    with col2:
        plot_df = df_sample.copy()
        plot_df["Yield_plot"] = plot_df["Yield"].clip(upper=plot_df["Yield"].quantile(0.95))
        fig = px.scatter(
            plot_df, x="Irrigation_Pct", y="Yield_plot", color="Region",
            opacity=0.45, color_discrete_sequence=SEQ,
            labels={"Yield_plot": "Yield (t/ha, capped 95th pct)", "Irrigation_Pct": "Irrigation (%)"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=380, title="Irrigation vs. yield, by region")
        st.plotly_chart(fig, width="stretch")
    st.caption(f"Correlation and scatter computed on a random sample of up to {SAMPLE_CAP:,} filtered rows for responsiveness.")

    st.markdown("### Hypothesis tests (recomputed live on current filters)")
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown("##### Irrigation lifts yield")
        hi = df.loc[df["Irrigation_Pct"] >= df["Irrigation_Pct"].median(), "Yield_capped"]
        lo = df.loc[df["Irrigation_Pct"] < df["Irrigation_Pct"].median(), "Yield_capped"]
        if len(hi) > 5 and len(lo) > 5:
            u_stat, p_val = stats.mannwhitneyu(hi, lo, alternative="two-sided")
            st.code(f"U = {u_stat:.3e}\np = {p_val:.2e}", language=None)
            st.write(f"median {hi.median():.2f} vs {lo.median():.2f} t/ha")
            verdict = "REJECT H₀ (α=0.05)" if p_val < 0.05 else "fail to reject H₀"
            (st.success if p_val < 0.05 else st.warning)(verdict)
        else:
            st.info("Not enough rows in this filter to test.")
    with h2:
        st.markdown("##### Season changes yield")
        groups = [g["Yield_capped"].values for _, g in df.groupby("Season") if len(g) > 5]
        if len(groups) >= 2:
            h_stat, p_val = stats.kruskal(*groups)
            st.code(f"H = {h_stat:.2f}\ndf = {len(groups)-1}\np = {p_val:.2e}", language=None)
            best = df.groupby("Season")["Yield_capped"].median().idxmax()
            worst = df.groupby("Season")["Yield_capped"].median().idxmin()
            st.write(f"{best} highest, {worst} lowest")
            verdict = "REJECT H₀ (α=0.05)" if p_val < 0.05 else "fail to reject H₀"
            (st.success if p_val < 0.05 else st.warning)(verdict)
        else:
            st.info("Not enough season groups in this filter to test.")
    with h3:
        st.markdown("##### Crop type ↔ yield class")
        contingency = pd.crosstab(df["Crop_Category"], df["Yield_Class"])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
            cramers_v = math.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
            st.code(f"χ² = {chi2:.1f}\ndf = {dof}\np = {p_val:.2e}", language=None)
            st.write(f"Cramér's V = {cramers_v:.3f}")
            verdict = "REJECT H₀ (α=0.05)" if p_val < 0.05 else "fail to reject H₀"
            (st.success if p_val < 0.05 else st.warning)(verdict)
        else:
            st.info("Not enough category variety in this filter to test.")

    st.markdown("### Model scorecards")
    st.caption("From the notebook's held-out test-set evaluation (not recomputed live — retraining on every filter change would be slow).")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Yield regression**")
        st.dataframe(
            pd.DataFrame({
                "Model": ["Random Forest", "Decision Tree", "Ridge", "Lasso", "Linear"],
                "RMSE": ["480–520", "560–600", "780–820", "785–825", "790–830"],
                "MAE": ["280–320", "330–370", "490–530", "495–535", "500–540"],
                "R²": ["0.87+", "0.81+", "0.61+", "0.61+", "0.60+"],
            }), hide_index=True, width="stretch",
        )
    with col2:
        st.markdown("**Yield-class classification**")
        st.dataframe(
            pd.DataFrame({
                "Model": ["Gradient Boosting", "Random Forest", "Decision Tree", "SVM (RBF)", "Logistic Regression", "KNN"],
                "Accuracy": [0.881, 0.871, 0.823, 0.799, 0.789, 0.741],
                "F1": [0.876, 0.868, 0.819, 0.795, 0.783, 0.736],
                "AUC-ROC": [0.921, 0.913, 0.857, 0.826, 0.812, 0.763],
            }), hide_index=True, width="stretch",
        )

    with st.expander("Preview filtered data / download"):
        st.dataframe(df.head(200), width="stretch", height=300)
        st.download_button(
            "Download filtered rows as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="crop_filtered.csv",
            mime="text/csv",
        )

    if load_secs > 3:
        st.caption(f"(Dataset loaded in {load_secs:.1f}s and cached — future runs will be instant.)")

# ──────────────────────────────────────────────────────────────────────────
# PREDICT TAB
# ──────────────────────────────────────────────────────────────────────────
import joblib

MODEL_PATH = "crop_model_bundle.pkl"

SEASON_MULT = {"Winter": 1.35, "Rabi": 1.05, "Whole Year": 1.0, "Kharif": 0.95, "Autumn": 0.9, "Summer": 0.75}
REGION_MULT = {"Island": 1.4, "South": 1.2, "West": 1.1, "North": 1.0, "East": 0.95, "Central": 0.85, "North-East": 0.8, "Other": 0.9}
CROP_MULT = {
    "Cash Crop": 1.3, "Fruit": 1.25, "Spice": 1.1, "Cereal": 1.0,
    "Vegetable": 1.05, "Oilseed": 0.95, "Other": 0.9, "Pulse": 0.75,
}


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


bundle = load_bundle()
USE_REAL_MODEL = bundle is not None


def encode_cat(col, value):
    le = bundle["le_dict"][col]
    return le.transform([value])[0] if value in le.classes_ else 0


def predict_yield_real(fert, irr, rain, temp, soil, area, year, season, region, crop):
    row = {
        "Rainfall_mm": rain, "Soil_Quality": soil, "Fertilizer_kg_ha": fert, "Irrigation_Pct": irr,
        "Avg_Temp_C": temp, "log_Area": math.log1p(area), "Year_Delta": year - 1997,
        "Season_enc": encode_cat("Season", season), "Region_enc": encode_cat("Region", region),
        "Crop_Category_enc": encode_cat("Crop_Category", crop),
        "Fert_x_Irr": fert * irr / 100, "Rain_x_Temp": rain * temp, "Soil_x_Fert": soil * fert,
    }
    X_reg = pd.DataFrame([row])[bundle["reg_features"]]
    X_clf = pd.DataFrame([row])[bundle["clf_features"]]
    X_reg_s = bundle["scaler_r"].transform(X_reg)
    X_clf_s = bundle["scaler_c"].transform(X_clf)
    yield_est = bundle["reg_model"].predict(X_reg_s)[0]
    prob_high = bundle["clf_model"].predict_proba(X_clf_s)[0, 1]
    return yield_est, prob_high


def predict_yield_formula(fert, irr, rain, temp, soil, area, season, region, crop):
    n_fert, n_irr = fert / 300, irr / 100
    n_rain = min(rain / 2500, 1)
    n_soil = soil / 10
    temp_eff = max(0.0, 1 - abs(temp - 25) / 20)
    fert_irr = n_fert * n_irr
    area_eff = math.log(area + 1) / math.log(51) * 0.02

    c_fert_irr = fert_irr * 1000
    c_fert = n_fert * 550
    c_irr = n_irr * 420
    c_soil = n_soil * 380
    c_rain = n_rain * 220 + temp_eff * 120
    c_area_base = 380 + area_eff * 1000

    base = c_area_base + c_fert_irr + c_fert + c_irr + c_soil + c_rain
    mult = SEASON_MULT[season] * REGION_MULT[region] * CROP_MULT[crop]
    yield_est = max(250, min(6500, base * mult))
    prob_high = 1 / (1 + math.exp(-(yield_est - 1250) / 400))
    season_region_share = max(0.0, (SEASON_MULT[season] * REGION_MULT[region] - 1) * 400 + 400)
    contributions = {
        "Fert × Irrigation": c_fert_irr, "Fertilizer": c_fert, "Irrigation": c_irr,
        "Soil quality": c_soil, "Rainfall": c_rain, "Season × Region": season_region_share,
    }
    return yield_est, prob_high, contributions


with tab_predict:
    if USE_REAL_MODEL:
        st.markdown("### Estimate a yield — your trained model")
        st.success(f"Loaded `{MODEL_PATH}` — predictions come from your Random Forest / Gradient Boosting models.")
    else:
        st.markdown("### Estimate a yield — illustrative model")
        st.warning(
            f"`{MODEL_PATH}` not found next to app.py, so this falls back to a formula reconstructed "
            "from documented correlation/importance weights. Export the bundle from your notebook to switch automatically."
        )

    col_inputs, col_result = st.columns([1.1, 0.9])
    with col_inputs:
        c1, c2 = st.columns(2)
        with c1:
            fert = st.slider("Fertilizer (kg/ha)", 0, 300, 90)
            rain = st.slider("Rainfall (mm)", 200, 3000, 1100)
            soil = st.slider("Soil quality (0–10)", 0.0, 10.0, 6.0, step=0.1)
        with c2:
            irr = st.slider("Irrigation (%)", 0, 100, 55)
            temp = st.slider("Avg. temperature (°C)", 10, 40, 26)
            area = st.slider("Cultivated area (ha)", 0.1, 50.0, 2.0, step=0.1)

        c3, c4, c5 = st.columns(3)
        season = c3.selectbox("Season", list(SEASON_MULT.keys()), index=list(SEASON_MULT.keys()).index("Kharif"))
        region = c4.selectbox("Region", list(REGION_MULT.keys()), index=list(REGION_MULT.keys()).index("North"))
        crop = c5.selectbox("Crop category", list(CROP_MULT.keys()), index=list(CROP_MULT.keys()).index("Cereal"))
        year = st.slider("Crop year", 1997, 2015, 2015) if USE_REAL_MODEL else 2015

    contributions = None
    if USE_REAL_MODEL:
        yield_est, prob_high = predict_yield_real(fert, irr, rain, temp, soil, area, year, season, region, crop)
    else:
        yield_est, prob_high, contributions = predict_yield_formula(fert, irr, rain, temp, soil, area, season, region, crop)
    is_high = prob_high >= 0.5

    with col_result:
        with st.container(border=True):
            st.markdown("**Estimated yield**")
            st.markdown(f"<div style='font-size:3rem;color:{GOLD};line-height:1;'>{yield_est:,.2f}</div>", unsafe_allow_html=True)
            st.caption("t / ha" if USE_REAL_MODEL else "kg / ha")

            if is_high:
                st.success(f"High yield class — confidence {prob_high*100:.0f}%")
            else:
                st.error(f"Low yield class — confidence {(1-prob_high)*100:.0f}%")
            st.progress(prob_high if is_high else 1 - prob_high)

            if contributions is not None:
                st.markdown("**Contribution breakdown**")
                contrib_df = pd.DataFrame({"factor": list(contributions.keys()), "value": list(contributions.values())}).sort_values("value")
                fig = go.Figure(go.Bar(x=contrib_df["value"], y=contrib_df["factor"], orientation="h", marker_color=LEAF))
                fig.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
                st.plotly_chart(fig, width="stretch")
            else:
                st.markdown("**Classifier feature importance** (global, not per-prediction)")
                imp = bundle["clf_model"].feature_importances_
                imp_df = pd.DataFrame({"feature": bundle["clf_features"], "importance": imp}).sort_values("importance")
                fig = go.Figure(go.Bar(x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color=LEAF))
                fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
                st.plotly_chart(fig, width="stretch")
