import streamlit as st
import pandas as pd
import plotly.express as px
from src.tracker import EvalTracker

st.set_page_config(page_title="LLM Eval Dashboard", layout="wide")

st.title("📊 LLM Evaluation Framework")
st.markdown("Track model regressions and metric drift over time.")

tracker = EvalTracker(storage_dir="eval_runs")
runs = tracker.load_all_runs()

if not runs:
    st.warning("No evaluation runs found. Run `example_run.py` to generate data.")
    st.stop()

# Prepare Data
data = []
for r in runs:
    row = {
        "timestamp": r["timestamp"],
        "model_version": r["model_version"],
        "dataset_name": r["dataset_name"],
    }
    # flatten metrics
    row.update(r["aggregate_metrics"])
    data.append(row)

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sidebar Filters
st.sidebar.header("Filters")
selected_dataset = st.sidebar.selectbox("Dataset", df["dataset_name"].unique())

df_filtered = df[df["dataset_name"] == selected_dataset]

# KPIs
st.subheader(f"Latest Run: {df_filtered.iloc[-1]['model_version']}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Exact Match", f"{df_filtered.iloc[-1]['avg_exact_match']:.2f}")
col2.metric("Valid JSON", f"{df_filtered.iloc[-1]['avg_valid_json']:.2f}")
col3.metric("Relevance", f"{df_filtered.iloc[-1]['avg_relevance']:.2f}")
col4.metric("Latency (s)", f"{df_filtered.iloc[-1]['avg_latency_sec']:.2f}")

st.divider()

# Drift Charts
st.subheader("Metric Regressions Over Time")

metrics_to_plot = ["avg_exact_match", "avg_valid_json", "avg_relevance"]
fig = px.line(
    df_filtered, 
    x="timestamp", 
    y=metrics_to_plot, 
    markers=True,
    text="model_version",
    title="Quality Metrics Trend"
)
fig.update_traces(textposition="top right")
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(
    df_filtered, 
    x="timestamp", 
    y="avg_latency_sec", 
    markers=True,
    title="Latency Drift (Lower is better)",
    color_discrete_sequence=["red"]
)
st.plotly_chart(fig2, use_container_width=True)

# Detail View
st.subheader("Raw Run Data")
st.dataframe(df_filtered)
