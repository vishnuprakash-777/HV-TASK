"""
Patient Longitudinal Health Intelligence — Streamlit Dashboard
==============================================================
Bonus deliverable for the AI Engineer take-home task.

A clinical-triage dashboard over the longitudinal blood-test analysis:
  - Population overview  : cohort-level health signals
  - Risk ranking         : every patient sorted by urgency, with filters
  - Patient explorer     : one patient's trends, burden, and LLM summary

Run with:
    pip install streamlit pandas numpy matplotlib
    streamlit run dashboard.py

Expects two files in the same folder:
    patient_summary.csv      (produced by analysis.ipynb, Section 5)
    patient_health_data.csv  (the original dataset)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ======================================================================
# 1. CONFIGURATION
# ======================================================================
SUMMARY_CSV = "patient_summary.csv"
RAW_CSV     = "patient_health_data.csv"

st.set_page_config(page_title="Patient Health Intelligence",
                   page_icon="HV", layout="wide")

# Urgency ranking - keep identical to the notebook's Section 2.
URGENCY = {
    "steadily_worsening": 7, "new_problem_emerging": 6, "relapsing": 5,
    "approaching_risk": 4, "chronically_abnormal_stable": 3,
    "recovering": 2, "steadily_improving": 1, "stable_normal": 0,
    "insufficient_data": -1,
}
URGENT_LABELS = {"steadily_worsening", "new_problem_emerging", "relapsing"}

TRAJ_COLOR = {
    "steadily_worsening": "#c0392b", "new_problem_emerging": "#e74c3c",
    "relapsing": "#e67e22", "approaching_risk": "#f39c12",
    "chronically_abnormal_stable": "#b7950b", "recovering": "#2980b9",
    "steadily_improving": "#27ae60", "stable_normal": "#7f8c8d",
    "insufficient_data": "#bdc3c7",
}
LABEL_HELP = {
    "steadily_worsening": "Consistent deterioration across multiple visits.",
    "new_problem_emerging": "Was normal in earlier visits, abnormal recently.",
    "relapsing": "Was normal/improved, now abnormal again.",
    "approaching_risk": "Still normal but drifting toward the boundary.",
    "chronically_abnormal_stable": "Persistently abnormal but not getting worse.",
    "recovering": "Was abnormal, now improving.",
    "steadily_improving": "Consistent improvement across multiple visits.",
    "stable_normal": "Within normal range with no meaningful trend.",
    "insufficient_data": "Too few visits to classify reliably.",
}
PLAIN_NAME = {
    "creatinine": "Creatinine", "uric_acid": "Uric Acid",
    "fasting_glucose": "Fasting Glucose", "hba1c": "HbA1c",
    "ldl": "LDL Cholesterol", "hdl": "HDL Cholesterol",
    "triglycerides": "Triglycerides", "sgot": "SGOT", "sgpt": "SGPT",
    "total_bilirubin": "Total Bilirubin", "total_protein": "Total Protein",
}
def nice(p): return PLAIN_NAME.get(p, p)


# ======================================================================
# 2. DATA LOADING  (cached - files are read once, not on every click)
# ======================================================================
@st.cache_data
def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df["param_trends_dict"]       = df["param_trends"].apply(json.loads)
    df["current_abnormal_params"] = df["current_abnormal_params"].fillna("")
    df["urgency_score"]           = df["worst_trajectory"].map(URGENCY).fillna(-1)
    return df


@st.cache_data
def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV, parse_dates=["date_of_test"])


# ======================================================================
# 3. CHART HELPERS
# ======================================================================
def plot_param(raw: pd.DataFrame, pid: str, param: str):
    """One marker's value over time, with the normal range shaded."""
    s = (raw[(raw.patient_id == pid) & (raw.param_name == param)]
         .sort_values("date_of_test"))
    if s.empty:
        return None
    low, high = s["low_range"].iloc[0], s["high_range"].iloc[0]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axhspan(low, high, color="#27ae60", alpha=0.12)
    ax.plot(s["date_of_test"], s["result"], "-o", color="#2c3e50",
            markersize=4, linewidth=1.4)

    abn = (s["result"] < low) | (s["result"] > high)
    if abn.any():
        ax.scatter(s.loc[abn, "date_of_test"], s.loc[abn, "result"],
                   color="#c0392b", zorder=5, s=45, label="out of range")
        ax.legend(fontsize=7, loc="best")

    ax.set_title(nice(param), fontsize=10, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel(s["unit"].iloc[0], fontsize=8)
    ax.tick_params(labelsize=7)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def plot_burden(raw: pd.DataFrame, pid: str):
    """Number of abnormal markers at each visit - the overall health burden."""
    s = raw[raw.patient_id == pid].copy()
    if s.empty:
        return None
    s["abnormal"] = (s["result"] < s["low_range"]) | (s["result"] > s["high_range"])
    by_visit = s.groupby("date_of_test")["abnormal"].sum().sort_index()

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.fill_between(by_visit.index, by_visit.values, color="#c0392b", alpha=0.12)
    ax.plot(by_visit.index, by_visit.values, "-o", color="#c0392b",
            markersize=5, linewidth=1.6)
    ax.set_title("Health burden over time (abnormal markers per visit)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("abnormal markers", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=7)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def show(fig):
    """Render a matplotlib figure and free its memory."""
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


# ======================================================================
# 4. LOAD DATA  (fail clearly if the files are missing)
# ======================================================================
try:
    summary = load_summary()
    raw     = load_raw()
except FileNotFoundError as e:
    st.error(f"Could not load a required data file: `{e.filename}`\n\n"
             "Place patient_summary.csv and patient_health_data.csv "
             "in the same folder as this script, then reload.")
    st.stop()


# ======================================================================
# 5. PAGE: POPULATION OVERVIEW
# ======================================================================
def render_overview():
    st.title("Population Overview")
    st.caption("Cohort-level view of where the patient population stands today.")

    pct_urgent = summary["worst_trajectory"].isin(URGENT_LABELS).mean() * 100
    n_abnormal = (summary["current_abnormal_params"].str.len() > 0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total patients", f"{len(summary):,}")
    c2.metric("In an urgent trajectory", f"{pct_urgent:.0f}%")
    c3.metric("1+ abnormal marker now", f"{n_abnormal:,}")
    c4.metric("Burden worsening",
              f"{(summary.burden_trend == 'worsening').sum():,}")

    st.divider()
    st.subheader("Worst trajectory across the population")
    st.caption("Each patient is counted once, by their most urgent parameter.")
    dist = summary["worst_trajectory"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(dist.index, dist.values,
           color=[TRAJ_COLOR.get(t, "#888") for t in dist.index])
    ax.set_ylabel("patients")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    show(fig)

    st.subheader("Overall health-burden trend")
    st.bar_chart(summary["burden_trend"].value_counts())

    with st.expander("What do the trajectory labels mean?"):
        for label, meaning in LABEL_HELP.items():
            st.markdown(f"**{label}** - {meaning}")


# ======================================================================
# 6. PAGE: RISK RANKING
# ======================================================================
def render_ranking():
    st.title("Patient Risk Ranking")
    st.caption("Every patient sorted most to least urgent. "
               "Use the filters to focus a cohort, then export the result.")

    f1, f2, f3 = st.columns(3)
    traj_filter   = f1.multiselect("Worst trajectory",
                                   sorted(summary["worst_trajectory"].unique()))
    burden_filter = f2.multiselect("Burden trend",
                                   sorted(summary["burden_trend"].unique()))
    gender_filter = f3.multiselect("Gender",
                                   sorted(summary["gender"].unique()))

    view = summary.copy()
    if traj_filter:
        view = view[view["worst_trajectory"].isin(traj_filter)]
    if burden_filter:
        view = view[view["burden_trend"].isin(burden_filter)]
    if gender_filter:
        view = view[view["gender"].isin(gender_filter)]
    view = view.sort_values("urgency_score", ascending=False)

    if view.empty:
        st.warning("No patients match the selected filters.")
        return

    st.write(f"Showing {len(view):,} of {len(summary):,} patients")
    table = view[["patient_id", "gender", "age_band", "total_visits",
                  "worst_trajectory", "burden_trend", "current_abnormal_params"]]
    st.dataframe(
        table, use_container_width=True, hide_index=True,
        column_config={
            "patient_id": "Patient", "gender": "Gender", "age_band": "Age band",
            "total_visits": "Visits", "worst_trajectory": "Worst trajectory",
            "burden_trend": "Burden trend",
            "current_abnormal_params": "Abnormal now",
        },
    )
    st.download_button("Download these results as CSV",
                       table.to_csv(index=False).encode(),
                       "risk_ranking.csv", "text/csv")


# ======================================================================
# 7. PAGE: PATIENT EXPLORER
# ======================================================================
def render_explorer():
    st.title("Patient Explorer")

    pid = st.sidebar.selectbox("Select patient",
                               sorted(summary["patient_id"].unique()))
    row = summary[summary.patient_id == pid].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Gender", row["gender"])
    c2.metric("Age band", row["age_band"])
    c3.metric("Total visits", int(row["total_visits"]))
    c4.metric("Worst trajectory", row["worst_trajectory"])
    c5.metric("Burden trend", row["burden_trend"])

    st.subheader("Clinical summary")
    st.info(row["llm_summary"])

    st.subheader("Overall health burden")
    show(plot_burden(raw, pid))

    st.subheader("Per-parameter trajectory")
    trends_tbl = (pd.DataFrame(
        [{"Parameter": nice(k), "Trajectory": v}
         for k, v in row["param_trends_dict"].items()])
        .sort_values("Trajectory"))

    def _colour(val):
        return f"background-color: {TRAJ_COLOR.get(val, '#ffffff')}; color: white"
    # Styler.map (pandas >= 2.1); the old name .applymap was removed in 3.0.
    st.dataframe(trends_tbl.style.map(_colour, subset=["Trajectory"]),
                 use_container_width=True, hide_index=True)

    abn = row["current_abnormal_params"]
    st.caption(f"Currently outside normal range: {abn if abn else 'none'}")

    st.subheader("Trend charts")
    params = sorted(row["param_trends_dict"].keys())
    cols = st.columns(3)
    for i, param in enumerate(params):
        with cols[i % 3]:
            show(plot_param(raw, pid, param))


# ======================================================================
# 8. NAVIGATION
# ======================================================================
st.sidebar.title("Health Intelligence")
page = st.sidebar.radio("View", ["Population overview", "Risk ranking",
                                 "Patient explorer"])
st.sidebar.caption(f"{len(summary):,} patients loaded")

if page == "Population overview":
    render_overview()
elif page == "Risk ranking":
    render_ranking()
else:
    render_explorer()
