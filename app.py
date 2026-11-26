from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt
st.set_page_config(
    page_title="PipeGuard AI",
    page_icon="💧",
    layout="wide"
)

@st.cache_data
def load_data():
    data_path = (
        Path(__file__).resolve().parent
        / "DATA"
        / "pipeguard_app_data.csv.gz"
    )
    return pd.read_csv(data_path)

df = load_data()


# ==========================================
# PIPEGUARD AI — INFRASTRUCTURE RISK DASHBOARD
# ==========================================

# Website title
st.title("💧 PipeGuard AI")

st.subheader("Water Infrastructure Risk Intelligence")

st.caption(
    "Pipeline Failure Prediction & Maintenance Prioritization"
)

st.info(
    "Research prototype based on synthetic/hybrid data. "
    "Risk scores are experimental and are not calibrated "
    "real-world failure probabilities."
)

st.divider()

# Main dashboard metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Pipelines",
        f"{df['pipe_id'].nunique():,}"
    )

with col2:
    st.metric(
        "Inspection Priorities",
        f"{df['top_5pct_priority'].sum():,}"
    )

with col3:
    st.metric(
        "Average Pipeline Age",
        f"{df['pipe_age_years'].mean():.1f} years"
    )

with col4:
    st.metric(
        "Average Model Score",
        f"{df['risk_score'].mean():.4f}"
    )

st.divider()

# Dashboard visualizations
left, right = st.columns(2)

with left:
    st.subheader("Pipeline Materials")

    material_counts = (
        df["pipe_material"]
        .value_counts()
    )

    st.bar_chart(material_counts)

with right:
    st.subheader("Risk Score Distribution")

    risk_chart = (
        alt.Chart(df)
        .mark_bar(color="#4A90C2")
        .encode(
            x=alt.X(
                "risk_score:Q",
                bin=alt.Bin(step=0.05, extent=[0, 1]),
                title="Experimental Risk Score"
            ),
            y=alt.Y(
                "count():Q",
                title="Number of Pipelines"
            ),
            tooltip=[
                alt.Tooltip("count():Q", title="Pipelines")
            ]
        )
        .properties(height=340)
    )

    st.altair_chart(
        risk_chart,
        use_container_width=True
    )
    
    
    
# ==========================================
# PIPEGUARD AI — PIPELINE EXPLORER
# ==========================================

st.divider()

st.header("🔎 Pipeline Explorer")

st.write(
    "Search and explore pipeline assets using the "
    "2025 assessment dataset."
)

# Interactive filters
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_materials = st.multiselect(
        "Filter by Pipe Material",
        options=sorted(df["pipe_material"].dropna().unique())
    )

with filter_col2:
    search_id = st.text_input(
        "Search Pipeline ID",
        placeholder="Enter a pipeline ID..."
    )

# Apply filters
filtered_df = df.copy()

if selected_materials:
    filtered_df = filtered_df[
        filtered_df["pipe_material"].isin(selected_materials)
    ]

if search_id:
    filtered_df = filtered_df[
        filtered_df["pipe_id"].astype(str).str.contains(
            search_id, regex=False
        )
    ]

st.write(f"Showing {len(filtered_df):,} pipelines")

# Display filtered results
explorer_columns = [
    "pipe_id",
    "pipe_material",
    "pipe_age_years",
    "diameter_mm",
    "condition_score",
    "risk_score",
    "inspection_rank"
]

st.dataframe(
    filtered_df[explorer_columns],
    hide_index=True,
    use_container_width=True
)

# Examine an individual pipeline
if not filtered_df.empty:

    selected_id = st.selectbox(
        "Select a pipeline to inspect",
        options=filtered_df["pipe_id"].tolist()
    )

    pipeline = filtered_df[
        filtered_df["pipe_id"] == selected_id
    ].iloc[0]

    st.subheader(f"Pipeline #{selected_id}")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Pipeline Age",
        f"{pipeline['pipe_age_years']} years"
    )

    metric2.metric(
        "Experimental Risk Score",
        f"{pipeline['risk_score']:.4f}"
    )

    metric3.metric(
        "Inspection Rank",
        f"{pipeline['inspection_rank']:,}"
    )

    if pipeline["top_5pct_priority"]:
        st.warning(
            "This pipeline is in the highest-ranked "
            "5% for inspection prioritization."
        )
    else:
        st.info(
            "This pipeline is outside the highest-ranked "
            "5% inspection group."
        )

    st.subheader("Pipeline Characteristics")

    detail_columns = [
        "pipe_material",
        "diameter_mm",
        "pipe_length_m",
        "pipe_age_years",
        "lining_material",
        "burial_depth_m",
        "soil_type",
        "condition_score",
        "previous_break_count",
        "inspections_prev_3y"
    ]

    st.dataframe(
        pipeline[detail_columns].to_frame("Value"),
        use_container_width=True
    )

else:
    st.warning("No pipelines match the selected filters.")


# ==========================================
# PIPEGUARD AI — INSPECTION PRIORITIZATION
# ==========================================

import numpy as np

st.divider()

st.header("📋 Inspection Prioritization")

st.write(
    "Prioritize pipeline inspections using experimental "
    "model risk rankings and available inspection capacity."
)

# Select inspection capacity
inspection_percentage = st.slider(
    "Inspection Capacity (% of Pipelines)",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

# Calculate how many pipelines can be inspected
inspection_count = int(
    np.ceil(len(df) * inspection_percentage / 100)
)

# Select highest-ranked pipelines
priority_df = (
    df.sort_values("inspection_rank")
    .head(inspection_count)
    .copy()
)

# Display planning metrics
p1, p2, p3 = st.columns(3)

p1.metric(
    "Pipelines to Inspect",
    f"{inspection_count:,}"
)

p2.metric(
    "Inspection Capacity",
    f"{inspection_percentage}%"
)

p3.metric(
    "Average Selected Risk Score",
    f"{priority_df['risk_score'].mean():.4f}"
)

# Show priority list
st.subheader("Recommended Inspection List")

priority_columns = [
    "inspection_rank",
    "pipe_id",
    "pipe_material",
    "pipe_age_years",
    "condition_score",
    "previous_break_count",
    "risk_score"
]

st.dataframe(
    priority_df[priority_columns],
    hide_index=True,
    use_container_width=True
)

# Download the selected inspection list
st.download_button(
    label="📥 Download Inspection List",
    data=priority_df[priority_columns].to_csv(
        index=False
    ),
    file_name="PipeGuard_Inspection_Priorities.csv",
    mime="text/csv"
)

st.caption(
    "Rankings are based on the 2025 retrospective "
    "assessment dataset and are intended for "
    "research demonstration, not operational decisions."
)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    