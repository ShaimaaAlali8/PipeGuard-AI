
from pathlib import Path
import math

import altair as alt
import joblib
import pandas as pd
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PipeGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 2. LOAD APPLICATION DATA AND MODEL
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        BASE_DIR / "DATA" / "pipeguard_app_data.csv.gz"
    )


@st.cache_resource
def load_prediction_model():
    model = joblib.load(
        BASE_DIR / "models" / "pipeguard_gradient_boosting.joblib"
    )
    features = joblib.load(
        BASE_DIR / "models" / "feature_columns.joblib"
    )
    return model, features


df = load_data()


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:
    st.title("💧 PipeGuard AI")
    st.caption("Infrastructure Risk Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "📊 Risk Dashboard",
            "🔎 Pipeline Explorer",
            "📋 Inspection Prioritization",
            "🧠 Risk Prediction",
            "📈 Model Information",
        ],
    )

    st.divider()

    st.caption("Assessment period: 2025")
    st.caption("Selected model: Gradient Boosting")
    st.caption("Research prototype")


# ==========================================
# 4. COMMON WEBSITE HEADER
# ==========================================

st.title("PipeGuard AI")

st.caption(
    "Water Distribution Pipeline Failure Prediction "
    "& Maintenance Prioritization"
)

st.info(
    "Research prototype using synthetic/hybrid data. "
    "Model scores are experimental ranking scores, "
    "not calibrated real-world failure probabilities."
)


# ==========================================
# 5. INFRASTRUCTURE RISK DASHBOARD
# ==========================================

def show_dashboard():
    st.header("📊 Infrastructure Risk Dashboard")

    # Bordered dashboard KPI cards
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        with st.container(border=True):
            st.metric(
                "Total Pipelines",
                f"{df['pipe_id'].nunique():,}"
            )

    with m2:
        with st.container(border=True):
            st.metric(
                "Inspection Priorities",
                f"{df['top_5pct_priority'].sum():,}"
            )

    with m3:
        with st.container(border=True):
            st.metric(
                "Average Pipeline Age",
                f"{df['pipe_age_years'].mean():.1f} years"
            )

    with m4:
        with st.container(border=True):
            st.metric(
                "Average Model Score",
                f"{df['risk_score'].mean():.4f}"
            )

    st.divider()

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

        chart = (
            alt.Chart(df)
            .mark_bar(color="#25C5BA")
            .encode(
                x=alt.X(
                    "risk_score:Q",
                    bin=alt.Bin(
                        step=0.05,
                        extent=[0, 1]
                    ),
                    title="Experimental Risk Score",
                ),
                y=alt.Y(
                    "count():Q",
                    title="Number of Pipelines"
                ),
                tooltip=[
                    alt.Tooltip(
                        "count():Q",
                        title="Pipelines"
                    )
                ],
            )
            .properties(height=340)
        )

        st.altair_chart(
            chart,
            use_container_width=True
        )

    st.divider()

    st.subheader("Top 10 Inspection Priorities")

    columns = [
        "inspection_rank",
        "pipe_id",
        "pipe_material",
        "pipe_age_years",
        "condition_score",
        "risk_score",
    ]

    top_10 = (
        df.sort_values("inspection_rank")[columns]
        .head(10)
    )

    st.dataframe(
        top_10,
        hide_index=True,
        use_container_width=True,
    )


# ==========================================
# 6. PIPELINE EXPLORER
# ==========================================

def show_explorer():
    st.header("🔎 Pipeline Explorer")

    st.write(
        "Search pipeline records and inspect "
        "their characteristics and scores."
    )

    c1, c2 = st.columns(2)

    with c1:
        materials = st.multiselect(
            "Filter by Material",
            options=sorted(
                df["pipe_material"].dropna().unique()
            ),
        )

    with c2:
        search_id = st.text_input(
            "Search Pipeline ID",
            placeholder="Enter a pipeline ID"
        )

    filtered = df.copy()

    if materials:
        filtered = filtered[
            filtered["pipe_material"].isin(materials)
        ]

    if search_id:
        filtered = filtered[
            filtered["pipe_id"]
            .astype(str)
            .str.contains(search_id, regex=False)
        ]

    st.metric(
        "Matching Pipelines",
        f"{len(filtered):,}"
    )

    columns = [
        "pipe_id",
        "pipe_material",
        "pipe_age_years",
        "diameter_mm",
        "condition_score",
        "risk_score",
        "inspection_rank",
    ]

    st.dataframe(
        filtered[columns],
        hide_index=True,
        use_container_width=True
    )

    if filtered.empty:
        st.warning("No pipelines match these filters.")
        return

    st.divider()

    selected_id = st.selectbox(
        "Select a pipeline to inspect",
        options=filtered["pipe_id"].tolist()
    )

    pipeline = filtered.loc[
        filtered["pipe_id"] == selected_id
    ].iloc[0]

    st.subheader(f"Pipeline #{selected_id}")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Pipeline Age",
        f"{pipeline['pipe_age_years']} years"
    )

    m2.metric(
        "Experimental Risk Score",
        f"{pipeline['risk_score']:.4f}"
    )

    m3.metric(
        "Inspection Rank",
        f"{pipeline['inspection_rank']:,}"
    )

    if pipeline["top_5pct_priority"]:
        st.warning(
            "This pipeline is in the highest-ranked "
            "5% for inspection."
        )
    else:
        st.info(
            "This pipeline is outside the "
            "highest-ranked 5% group."
        )

    st.subheader("Pipeline Characteristics")

    details = [
        "pipe_material",
        "diameter_mm",
        "pipe_length_m",
        "pipe_age_years",
        "lining_material",
        "burial_depth_m",
        "soil_type",
        "condition_score",
        "previous_break_count",
        "inspections_prev_3y",
    ]

    st.dataframe(
        pipeline[details].to_frame("Value"),
        use_container_width=True
    )


# ==========================================
# 7. INSPECTION PRIORITIZATION
# ==========================================

def show_prioritization():
    st.header("📋 Inspection Prioritization")

    st.write(
        "Adjust inspection capacity to select "
        "the highest-ranked pipelines."
    )

    percentage = st.slider(
        "Inspection Capacity (% of Pipelines)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    inspection_count = math.ceil(
        len(df) * percentage / 100
    )

    selected = (
        df.sort_values("inspection_rank")
        .head(inspection_count)
        .copy()
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Pipelines to Inspect",
        f"{inspection_count:,}"
    )

    m2.metric(
        "Inspection Capacity",
        f"{percentage}%"
    )

    m3.metric(
        "Average Selected Score",
        f"{selected['risk_score'].mean():.4f}"
    )

    st.divider()

    st.subheader("Recommended Inspection List")

    columns = [
        "inspection_rank",
        "pipe_id",
        "pipe_material",
        "pipe_age_years",
        "condition_score",
        "previous_break_count",
        "risk_score",
    ]

    st.dataframe(
        selected[columns],
        hide_index=True,
        use_container_width=True
    )

    st.download_button(
        "📥 Download Inspection List",
        data=selected[columns].to_csv(index=False),
        file_name="PipeGuard_Inspection_Priorities.csv",
        mime="text/csv",
    )

    st.caption(
        "Retrospective 2025 rankings for "
        "research demonstration only."
    )


# ==========================================
# 8. INTERACTIVE RISK PREDICTION
# ==========================================

def show_prediction():
    st.header("🧠 Pipeline Risk Prediction")

    st.write(
        "Select a 2025 reference pipeline, "
        "adjust its characteristics, "
        "and calculate a new experimental score."
    )

    st.caption(
        "The reference values come from "
        "the 2025 assessment snapshot."
    )

    model, feature_columns = load_prediction_model()

    reference_id = st.selectbox(
        "Reference Pipeline ID",
        options=sorted(df["pipe_id"].unique())
    )

    reference = df.loc[
        df["pipe_id"] == reference_id
    ].iloc[0]

    categorical_features = {
        "pressure_zone",
        "pipe_material",
        "lining_material",
        "soil_type",
    }

    integer_features = {
        "diameter_mm",
        "pipe_age_years",
        "criticality_rating",
        "customers_served",
        "frost_days_prev_12m",
        "previous_break_count",
        "inspections_prev_3y",
        "years_since_last_inspection",
    }

    values = {}

    with st.form("pipeline_prediction_form"):
        st.subheader("Pipeline Input Features")

        left, right = st.columns(2)

        for index, column in enumerate(feature_columns):
            if column in (
                "has_previous_break",
                "years_since_last_break"
            ):
                continue

            label = column.replace("_", " ").title()

            with (left if index % 2 == 0 else right):
                if column in categorical_features:
                    choices = sorted(
                        df[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    selected_value = str(reference[column])

                    default_index = (
                        choices.index(selected_value)
                        if selected_value in choices
                        else 0
                    )

                    values[column] = st.selectbox(
                        label,
                        choices,
                        index=default_index,
                        key=f"prediction_{reference_id}_{column}",
                    )
                else:
                    default = reference[column]

                    if pd.isna(default):
                        default = pd.to_numeric(
                            df[column],
                            errors="coerce"
                        ).median()

                    if pd.isna(default):
                        default = 0

                    if column in integer_features:
                        values[column] = st.number_input(
                            label,
                            min_value=0,
                            value=max(
                                0,
                                int(round(float(default)))
                            ),
                            step=1,
                            key=f"prediction_{reference_id}_{column}",
                        )
                    else:
                        values[column] = st.number_input(
                            label,
                            min_value=0.0,
                            value=max(
                                0.0,
                                float(default)
                            ),
                            step=0.1,
                            format="%.2f",
                            key=f"prediction_{reference_id}_{column}",
                        )

        st.divider()

        st.subheader("Previous Break History")

        historical_recency = reference[
            "years_since_last_break"
        ]

        if pd.isna(historical_recency):
            historical_recency = df.loc[
                df["previous_break_count"] > 0,
                "years_since_last_break"
            ].median()

        if pd.isna(historical_recency):
            historical_recency = 1.0

        years_since_break = st.number_input(
            "Years Since Last Break",
            min_value=0.0,
            value=float(historical_recency),
            step=0.5,
            help=(
                "Only used when Previous Break Count "
                "is greater than zero."
            ),
            key=f"prediction_{reference_id}_years_since_last_break",
        )

        count = values["previous_break_count"]

        values["has_previous_break"] = int(count > 0)

        values["years_since_last_break"] = (
            float(years_since_break)
            if count > 0
            else float("nan")
        )

        submitted = st.form_submit_button(
            "🔍 Generate Risk Score",
            type="primary",
            use_container_width=True
        )

    if not submitted:
        return

    # Validate entered measurements
    if (
        values["avg_pressure_bar_prev_12m"]
        > values["max_pressure_bar_prev_12m"]
    ):
        st.error(
            "Average pressure cannot exceed maximum pressure."
        )
        return

    if (
        count > 0
        and years_since_break > values["pipe_age_years"]
    ):
        st.error(
            "Years since last break cannot exceed "
            "the pipeline's age."
        )
        return

    # Generate prediction with the saved model
    model_input = pd.DataFrame(
        [values],
        columns=feature_columns
    )

    score = float(
        model.predict_proba(model_input)[0, 1]
    )

    percentile = float(
        df["risk_score"].le(score).mean() * 100
    )

    top_5_cutoff = float(
        df.loc[
            df["top_5pct_priority"],
            "risk_score"
        ].min()
    )

    st.divider()

    st.subheader("Prediction Results")

    c1, c2 = st.columns(2)

    c1.metric(
        "Experimental Risk Score",
        f"{score:.4f}"
    )

    c2.metric(
        "2025 Reference Percentile",
        f"{percentile:.1f}%"
    )

    if score >= top_5_cutoff:
        st.warning(
            "This score meets the 2025 "
            "top-5% reference cutoff."
        )
    else:
        st.info(
            "This score is below the 2025 "
            "top-5% reference cutoff."
        )

    st.caption(
        "The score is not a calibrated real-world "
        "failure probability. Changing the form creates "
        "a hypothetical assessment, not a live forecast."
    )

    with st.expander("View Model Inputs"):
        st.dataframe(
            model_input.T.rename(
                columns={0: "Input Value"}
            ),
            use_container_width=True,
        )


# ==========================================
# 9. MODEL INFORMATION & PERFORMANCE
# ==========================================

def show_model_information():
    st.header("📈 Model Information & Performance")

    st.write(
        "How PipeGuard AI was trained, evaluated, "
        "and selected for experimental pipeline "
        "inspection prioritization."
    )

    st.info(
        "Results below come from the research notebook, "
        "using hybrid/synthetic data. They are not live "
        "operational performance measurements."
    )

    # Model overview
    st.subheader("Model Overview")

    a, b, c = st.columns(3)

    a.metric("Selected Model", "Gradient Boosting")
    b.metric("Input Features", "22")
    c.metric("Prediction Horizon", "12 months")

    st.write(
        "The target is `failure_next_12_months` "
        "(1 = failure during the following 12 months; "
        "0 = no recorded/simulated failure). "
        "Model inputs reflect the assessment date."
    )

    st.divider()

    # Chronological data split
    st.subheader("Chronological Data Split")

    split = pd.DataFrame({
        "Dataset": [
            "Training",
            "Validation",
            "Testing"
        ],
        "Assessment Years": [
            "2009–2023",
            "2024",
            "2025"
        ],
        "Records": [
            171879,
            14055,
            14209
        ],
    })

    st.dataframe(
        split,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Imputation, scaling, and encoding were fitted "
        "using training data. Validation was used for "
        "model selection; 2025 was reserved for "
        "final evaluation."
    )

    st.divider()

    # Three-model comparison
    st.subheader(
        "2024 Validation: Three-Model Comparison"
    )

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting"
        ],
        "Average Precision": [
            0.0290,
            0.0208,
            0.0324
        ],
        "Precision @ 0.5": [
            0.0148,
            0.0346,
            0.0000
        ],
        "Recall @ 0.5": [
            0.7614,
            0.1477,
            0.0000
        ],
        "F1 @ 0.5": [
            0.0290,
            0.0560,
            0.0000
        ],
        "Failures in Top 5%": [
            22,
            19,
            25
        ],
        "Top 5% Capture (%)": [
            25.00,
            21.59,
            28.41
        ],
    })

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:
        st.caption("Average Precision")

        ap_chart = (
            alt.Chart(comparison)
            .mark_bar(color="#25C5BA")
            .encode(
                x=alt.X(
                    "Model:N",
                    title=None,
                    sort="-y"
                ),
                y=alt.Y(
                    "Average Precision:Q",
                    title="Average Precision"
                ),
                tooltip=[
                    "Model:N",
                    alt.Tooltip(
                        "Average Precision:Q",
                        format=".4f"
                    )
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(
            ap_chart,
            use_container_width=True
        )

    with right:
        st.caption(
            "Failures Captured Among Highest-Ranked 5%"
        )

        capture_chart = (
            alt.Chart(comparison)
            .mark_bar(color="#25C5BA")
            .encode(
                x=alt.X(
                    "Model:N",
                    title=None,
                    sort="-y"
                ),
                y=alt.Y(
                    "Top 5% Capture (%):Q",
                    title="Capture rate (%)"
                ),
                tooltip=[
                    "Model:N",
                    "Top 5% Capture (%):Q"
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(
            capture_chart,
            use_container_width=True
        )

    st.warning(
        "Gradient Boosting detected no failures using "
        "a fixed 0.5 classification threshold in validation. "
        "Its advantage was in ranking pipelines, "
        "not binary classification at that threshold."
    )

    st.write(
        "Gradient Boosting was chosen because its "
        "validation Average Precision and top-5% failure "
        "capture were highest among the three models."
    )

    st.divider()

    # Walk-forward cross-validation
    st.subheader("Walk-Forward Cross-Validation")

    cv = pd.DataFrame({
        "Validation Year": [
            2021,
            2022,
            2023
        ],
        "Average Precision": [
            0.0289,
            0.0145,
            0.0273
        ],
        "Baseline AP": [
            0.0059,
            0.0055,
            0.0063
        ],
        "Top 5% Capture (%)": [
            26.92,
            14.86,
            21.84
        ],
    })

    st.dataframe(
        cv,
        hide_index=True,
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Mean CV Average Precision",
        "0.0235"
    )

    c2.metric(
        "Mean Top-5% Capture",
        "21.21%"
    )

    st.divider()

    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")

    tuned_params = pd.DataFrame({
        "Parameter": [
            "learning_rate",
            "max_depth",
            "n_estimators"
        ],
        "Selected Value": [
            "0.05",
            "3",
            "100"
        ],
    })

    st.dataframe(
        tuned_params,
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "GridSearchCV used historical walk-forward "
        "validation. The best tested configuration "
        "matched the original Gradient Boosting settings."
    )

    st.divider()

    # Final test evaluation
    st.subheader("Final Evaluation: 2025 Test Set")

    t1, t2, t3 = st.columns(3)

    t1.metric(
        "Average Precision",
        "0.0209"
    )

    t2.metric(
        "Baseline AP",
        "0.0065"
    )

    t3.metric(
        "Top-5% Lift",
        "3.65×"
    )

    final_test = pd.DataFrame({
        "Metric": [
            "Total test assessments",
            "Simulated failures",
            "Top-5% selected",
            "Failures captured",
            "Top-5% precision",
            "Failure capture rate",
        ],
        "Value": [
            "14,209",
            "93",
            "711",
            "17",
            "2.39%",
            "18.28%"
        ],
    })

    st.dataframe(
        final_test,
        hide_index=True,
        use_container_width=True
    )

    st.write(
        "Selecting the 711 highest-ranked 2025 "
        "assessments captured 17 of 93 simulated "
        "failures. This is a retrospective test of "
        "prioritization, not evidence of operational "
        "deployment performance."
    )

    st.divider()

    # Business purpose and limitations
    st.subheader("Business Purpose & Limitations")

    st.write(
        "The application demonstrates how asset managers "
        "and maintenance planners could allocate limited "
        "inspections using risk rankings."
    )

    st.warning(
        "Synthetic/hybrid records, simulated outcomes, "
        "and possible changes in real infrastructure "
        "conditions limit the findings. The model needs "
        "independent evaluation and probability calibration "
        "on verified utility data before being used for "
        "real maintenance decisions."
    )

    st.caption(
        "Notebook results shown here are fixed "
        "and are not recalculated live."
    )


# ==========================================
# 10. DISPLAY THE SELECTED PAGE
# ==========================================

if page == "📊 Risk Dashboard":
    show_dashboard()

elif page == "🔎 Pipeline Explorer":
    show_explorer()

elif page == "📋 Inspection Prioritization":
    show_prioritization()

elif page == "🧠 Risk Prediction":
    show_prediction()

elif page == "📈 Model Information":
    show_model_information()


# ==========================================
# 11. WEBSITE FOOTER
# ==========================================

st.divider()

st.caption(
    "PipeGuard AI | Water Infrastructure Risk Intelligence "
    "| Experimental Research Prototype"
)
