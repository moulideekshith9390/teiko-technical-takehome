import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analysis
import load_data


ROOT_DIR = Path(__file__).resolve().parent

DB_PATH = ROOT_DIR / "cell_counts.db"

STATISTICS_PATH = (
    ROOT_DIR
    / "outputs"
    / "statistical_results.csv"
)


st.set_page_config(
    page_title="Teiko Immune Cell Analysis",
    layout="wide",
)


def ensure_pipeline_outputs():
    """
    Make the dashboard self-contained when deployed.
    """

    if not DB_PATH.exists():
        load_data.main()

    if not STATISTICS_PATH.exists():
        analysis.main()


@st.cache_data
def load_frequency_data():

    connection = sqlite3.connect(
        DB_PATH
    )

    query = """
        SELECT
            s.project_id AS project,
            s.subject_id,
            s.condition,
            s.sex,
            s.treatment,
            s.response,
            sm.sample_id,
            sm.sample_type,
            sm.time_from_treatment_start,
            cc.cell_type AS population,
            cc.cell_count AS count,
            totals.total_count,
            100.0 * cc.cell_count /
                totals.total_count AS percentage

        FROM subjects AS s

        JOIN samples AS sm
            ON s.subject_id = sm.subject_id

        JOIN cell_counts AS cc
            ON sm.sample_id = cc.sample_id

        JOIN (
            SELECT
                sample_id,
                SUM(cell_count) AS total_count
            FROM cell_counts
            GROUP BY sample_id
        ) AS totals
            ON sm.sample_id = totals.sample_id;
    """

    data = pd.read_sql_query(
        query,
        connection,
    )

    connection.close()

    return data


ensure_pipeline_outputs()

data = load_frequency_data()

statistics = pd.read_csv(
    STATISTICS_PATH
)


st.title(
    "Immune Cell Population Analysis"
)

st.caption(
    "Interactive exploration of immune-cell "
    "frequencies, treatment response, and "
    "baseline clinical-trial subsets."
)


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.header(
    "Filters"
)

condition = st.sidebar.selectbox(
    "Condition",
    ["All"]
    + sorted(
        data["condition"]
        .dropna()
        .unique()
        .tolist()
    ),
)

treatment = st.sidebar.selectbox(
    "Treatment",
    ["All"]
    + sorted(
        data["treatment"]
        .dropna()
        .unique()
        .tolist()
    ),
)

sample_type = st.sidebar.selectbox(
    "Sample type",
    ["All"]
    + sorted(
        data["sample_type"]
        .dropna()
        .unique()
        .tolist()
    ),
)

population = st.sidebar.selectbox(
    "Population",
    sorted(
        data["population"]
        .unique()
        .tolist()
    ),
)


filtered = data.copy()

if condition != "All":
    filtered = filtered[
        filtered["condition"] == condition
    ]

if treatment != "All":
    filtered = filtered[
        filtered["treatment"] == treatment
    ]

if sample_type != "All":
    filtered = filtered[
        filtered["sample_type"] == sample_type
    ]

filtered = filtered[
    filtered["population"] == population
]


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

column1, column2, column3 = (
    st.columns(3)
)

column1.metric(
    "Samples",
    f"{filtered['sample_id'].nunique():,}",
)

column2.metric(
    "Subjects",
    f"{filtered['subject_id'].nunique():,}",
)

mean_frequency = (
    filtered["percentage"].mean()
    if not filtered.empty
    else None
)

column3.metric(
    "Mean frequency",
    (
        f"{mean_frequency:.2f}%"
        if mean_frequency is not None
        else "N/A"
    ),
)


st.subheader(
    "Filtered frequency distribution"
)

if filtered.empty:

    st.info(
        "No observations match the selected filters."
    )

else:

    figure, axis = plt.subplots(
        figsize=(8, 4.5)
    )

    axis.hist(
        filtered["percentage"],
        bins=30,
    )

    axis.set_xlabel(
        "Cell frequency (%)"
    )

    axis.set_ylabel(
        "Number of observations"
    )

    axis.set_title(
        population
        .replace("_", " ")
        .title()
    )

    figure.tight_layout()

    st.pyplot(
        figure
    )

    plt.close(
        figure
    )


# -------------------------------------------------------------------
# Part 3
# -------------------------------------------------------------------

st.subheader(
    "Miraclib response comparison"
)

response_data = data[
    (
        data["condition"].str.lower()
        == "melanoma"
    )
    &
    (
        data["treatment"].str.lower()
        == "miraclib"
    )
    &
    (
        data["sample_type"].str.lower()
        == "pbmc"
    )
].copy()


response_data[
    "response_label"
] = (
    response_data["response"]
    .str.lower()
    .map(
        {
            "yes": "Responder",
            "no": "Non-responder",
        }
    )
)


subject_level = (
    response_data
    .dropna(
        subset=["response_label"]
    )
    .groupby(
        [
            "subject_id",
            "response_label",
            "population",
        ],
        as_index=False,
    )["percentage"]
    .mean()
)


selected_population = st.selectbox(
    "Cell population for response comparison",
    sorted(
        subject_level[
            "population"
        ]
        .unique()
        .tolist()
    ),
)


plot_data = subject_level[
    subject_level["population"]
    == selected_population
]


responders = plot_data.loc[
    plot_data["response_label"]
    == "Responder",
    "percentage",
]


non_responders = plot_data.loc[
    plot_data["response_label"]
    == "Non-responder",
    "percentage",
]


figure, axis = plt.subplots(
    figsize=(7, 5)
)

axis.boxplot(
    [
        responders,
        non_responders,
    ],
    tick_labels=[
        "Responder",
        "Non-responder",
    ],
)

axis.set_ylabel(
    "Mean cell frequency per subject (%)"
)

axis.set_title(
    selected_population
    .replace("_", " ")
    .title()
)

figure.tight_layout()

st.pyplot(
    figure
)

plt.close(
    figure
)


stat_row = statistics[
    statistics["population"]
    == selected_population
]

if not stat_row.empty:

    row = stat_row.iloc[0]

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Responder mean",
        f"{row['responder_mean_pct']:.2f}%",
    )

    metric2.metric(
        "Non-responder mean",
        f"{row['non_responder_mean_pct']:.2f}%",
    )

    metric3.metric(
        "FDR-adjusted p",
        f"{row['adjusted_p_value']:.4f}",
    )

    metric4.metric(
        "Cohen's d",
        f"{row['cohens_d']:.3f}",
    )


st.write(
    "Statistical results"
)

st.dataframe(
    statistics[
        [
            "population",
            "responder_mean_pct",
            "non_responder_mean_pct",
            "p_value",
            "adjusted_p_value",
            "cohens_d",
            "significant_fdr_0_05",
        ]
    ],
    width="stretch",
)


# -------------------------------------------------------------------
# Part 4
# -------------------------------------------------------------------

st.subheader(
    "Baseline melanoma / miraclib / PBMC subset"
)


baseline = data[
    (
        data["condition"].str.lower()
        == "melanoma"
    )
    &
    (
        data["treatment"].str.lower()
        == "miraclib"
    )
    &
    (
        data["sample_type"].str.lower()
        == "pbmc"
    )
    &
    (
        data[
            "time_from_treatment_start"
        ] == 0
    )
][
    [
        "project",
        "subject_id",
        "response",
        "sex",
        "sample_id",
    ]
].drop_duplicates()


left_column, right_column = (
    st.columns(2)
)


with left_column:

    st.write(
        "Samples by project"
    )

    st.dataframe(
        baseline
        .groupby("project")
        .size()
        .rename("sample_count")
        .reset_index(),
        width="stretch",
    )

    st.write(
        "Subjects by response"
    )

    st.dataframe(
        baseline[
            [
                "subject_id",
                "response",
            ]
        ]
        .drop_duplicates()
        .groupby("response")
        .size()
        .rename("subject_count")
        .reset_index(),
        width="stretch",
    )


with right_column:

    st.write(
        "Subjects by sex"
    )

    st.dataframe(
        baseline[
            [
                "subject_id",
                "sex",
            ]
        ]
        .drop_duplicates()
        .groupby("sex")
        .size()
        .rename("subject_count")
        .reset_index(),
        width="stretch",
    )

    st.metric(
        "Baseline samples",
        f"{baseline['sample_id'].nunique():,}",
    )