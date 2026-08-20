import csv
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "cell_counts.db"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

FREQUENCY_OUTPUT = OUTPUT_DIR / "cell_frequencies.csv"
SUBJECT_FREQUENCY_OUTPUT = OUTPUT_DIR / "subject_level_frequencies.csv"
STATISTICS_OUTPUT = OUTPUT_DIR / "statistical_results.csv"

BASELINE_OUTPUT = OUTPUT_DIR / "baseline_melanoma_pbmc.csv"
PROJECT_COUNTS_OUTPUT = OUTPUT_DIR / "part4_project_counts.csv"
RESPONSE_COUNTS_OUTPUT = OUTPUT_DIR / "part4_response_counts.csv"
SEX_COUNTS_OUTPUT = OUTPUT_DIR / "part4_sex_counts.csv"

FORM_ANSWER_OUTPUT = OUTPUT_DIR / "form_answer.txt"


# -------------------------------------------------------------------
# Part 2
# -------------------------------------------------------------------

def calculate_cell_frequencies():
    """Calculate immune-cell frequency for every sample."""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run 'python load_data.py' first."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    query = """
        SELECT
            cc.sample_id AS sample,
            totals.total_count,
            cc.cell_type AS population,
            cc.cell_count AS count,
            100.0 * cc.cell_count / totals.total_count AS percentage
        FROM cell_counts AS cc
        JOIN (
            SELECT
                sample_id,
                SUM(cell_count) AS total_count
            FROM cell_counts
            GROUP BY sample_id
        ) AS totals
            ON cc.sample_id = totals.sample_id
        ORDER BY
            cc.sample_id,
            cc.cell_type;
    """

    rows = connection.execute(query).fetchall()
    connection.close()

    with FREQUENCY_OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.writer(output_file)

        writer.writerow(
            [
                "sample",
                "total_count",
                "population",
                "count",
                "percentage",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row["sample"],
                    row["total_count"],
                    row["population"],
                    row["count"],
                    round(row["percentage"], 4),
                ]
            )

    print("Part 2: Cell frequency analysis")
    print("--------------------------------")
    print(f"Rows generated: {len(rows):,}")
    print(f"Output: {FREQUENCY_OUTPUT}")


# -------------------------------------------------------------------
# Statistical helpers
# -------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction."""

    p_values = list(p_values)
    number_of_tests = len(p_values)

    ordered = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * number_of_tests
    running_minimum = 1.0

    for rank in range(number_of_tests, 0, -1):

        original_index, p_value = ordered[rank - 1]

        corrected = min(
            running_minimum,
            p_value * number_of_tests / rank,
            1.0,
        )

        adjusted[original_index] = corrected
        running_minimum = corrected

    return adjusted


def cohens_d(group_a, group_b):
    """Calculate Cohen's d effect size."""

    n_a = len(group_a)
    n_b = len(group_b)

    variance_a = group_a.var(ddof=1)
    variance_b = group_b.var(ddof=1)

    pooled_variance = (
        ((n_a - 1) * variance_a)
        + ((n_b - 1) * variance_b)
    ) / (n_a + n_b - 2)

    pooled_sd = pooled_variance ** 0.5

    if pooled_sd == 0:
        return 0.0

    return (
        group_a.mean() - group_b.mean()
    ) / pooled_sd


# -------------------------------------------------------------------
# Part 3
# -------------------------------------------------------------------

def analyze_responder_differences():
    """
    Compare cell frequencies between responders and non-responders
    among melanoma subjects treated with miraclib using PBMC samples.

    Repeated samples are aggregated to the subject level before
    hypothesis testing.
    """

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            s.subject_id,
            s.response,
            sm.sample_id,
            cc.cell_type,
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
            ON sm.sample_id = totals.sample_id
        WHERE
            LOWER(s.condition) = 'melanoma'
            AND LOWER(s.treatment) = 'miraclib'
            AND LOWER(sm.sample_type) = 'pbmc'
            AND LOWER(s.response) IN ('yes', 'no')
        ORDER BY
            s.subject_id,
            sm.sample_id,
            cc.cell_type;
    """

    sample_level = pd.read_sql_query(
        query,
        connection,
    )

    connection.close()

    if sample_level.empty:
        raise RuntimeError(
            "No samples matched the Part 3 filters."
        )

    sample_level["response"] = (
        sample_level["response"]
        .str.lower()
        .map(
            {
                "yes": "Responder",
                "no": "Non-responder",
            }
        )
    )

    subject_level = (
        sample_level
        .groupby(
            [
                "subject_id",
                "response",
                "cell_type",
            ],
            as_index=False,
        )["percentage"]
        .mean()
    )

    subject_level.to_csv(
        SUBJECT_FREQUENCY_OUTPUT,
        index=False,
    )

    results = []

    for cell_type in sorted(
        subject_level["cell_type"].unique()
    ):

        population_data = subject_level[
            subject_level["cell_type"] == cell_type
        ]

        responders = population_data.loc[
            population_data["response"] == "Responder",
            "percentage",
        ]

        non_responders = population_data.loc[
            population_data["response"] == "Non-responder",
            "percentage",
        ]

        test = stats.ttest_ind(
            responders,
            non_responders,
            equal_var=False,
            nan_policy="omit",
        )

        results.append(
            {
                "population": cell_type,
                "responder_n": len(responders),
                "non_responder_n": len(non_responders),
                "responder_mean_pct": responders.mean(),
                "non_responder_mean_pct": non_responders.mean(),
                "mean_difference_pct": (
                    responders.mean()
                    - non_responders.mean()
                ),
                "t_statistic": test.statistic,
                "p_value": test.pvalue,
                "cohens_d": cohens_d(
                    responders,
                    non_responders,
                ),
            }
        )

        figure, axis = plt.subplots(
            figsize=(6, 5)
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

        axis.set_title(
            f"{cell_type.replace('_', ' ').title()}: "
            "Responders vs Non-responders"
        )

        axis.set_xlabel(
            "Clinical response"
        )

        axis.set_ylabel(
            "Mean cell frequency per subject (%)"
        )

        figure.tight_layout()

        figure.savefig(
            FIGURE_DIR
            / f"{cell_type}_response_boxplot.png",
            dpi=300,
        )

        plt.close(figure)

    results_df = pd.DataFrame(results)

    results_df["adjusted_p_value"] = (
        benjamini_hochberg(
            results_df["p_value"]
        )
    )

    results_df["significant_fdr_0_05"] = (
        results_df["adjusted_p_value"] < 0.05
    )

    results_df = results_df[
        [
            "population",
            "responder_n",
            "non_responder_n",
            "responder_mean_pct",
            "non_responder_mean_pct",
            "mean_difference_pct",
            "t_statistic",
            "p_value",
            "adjusted_p_value",
            "cohens_d",
            "significant_fdr_0_05",
        ]
    ]

    results_df.to_csv(
        STATISTICS_OUTPUT,
        index=False,
    )

    print(
        "\nPart 3: Responder vs non-responder analysis"
    )
    print(
        "-------------------------------------------"
    )

    print(
        f"Samples matching filters: "
        f"{sample_level['sample_id'].nunique():,}"
    )

    print(
        f"Subjects matching filters: "
        f"{subject_level['subject_id'].nunique():,}"
    )

    print("\nStatistical results:")

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nStatistics saved to: "
        f"{STATISTICS_OUTPUT}"
    )

    print(
        f"Boxplots saved to: "
        f"{FIGURE_DIR}"
    )


# -------------------------------------------------------------------
# Part 4
# -------------------------------------------------------------------

def analyze_baseline_samples():
    """
    Analyze baseline melanoma PBMC samples from subjects
    treated with miraclib.
    """

    connection = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            s.project_id AS project,
            s.subject_id,
            s.response,
            s.sex,
            sm.sample_id,
            sm.sample_type,
            sm.time_from_treatment_start
        FROM subjects AS s
        JOIN samples AS sm
            ON s.subject_id = sm.subject_id
        WHERE
            LOWER(s.condition) = 'melanoma'
            AND LOWER(s.treatment) = 'miraclib'
            AND LOWER(sm.sample_type) = 'pbmc'
            AND sm.time_from_treatment_start = 0
        ORDER BY
            s.project_id,
            s.subject_id;
    """

    baseline = pd.read_sql_query(
        query,
        connection,
    )

    connection.close()

    if baseline.empty:
        raise RuntimeError(
            "No baseline samples matched Part 4 filters."
        )

    baseline.to_csv(
        BASELINE_OUTPUT,
        index=False,
    )

    project_counts = (
        baseline
        .groupby("project")
        .size()
        .rename("sample_count")
        .reset_index()
    )

    response_counts = (
        baseline[
            ["subject_id", "response"]
        ]
        .drop_duplicates()
        .groupby("response")
        .size()
        .rename("subject_count")
        .reset_index()
    )

    sex_counts = (
        baseline[
            ["subject_id", "sex"]
        ]
        .drop_duplicates()
        .groupby("sex")
        .size()
        .rename("subject_count")
        .reset_index()
    )

    project_counts.to_csv(
        PROJECT_COUNTS_OUTPUT,
        index=False,
    )

    response_counts.to_csv(
        RESPONSE_COUNTS_OUTPUT,
        index=False,
    )

    sex_counts.to_csv(
        SEX_COUNTS_OUTPUT,
        index=False,
    )

    print(
        "\nPart 4: Baseline melanoma PBMC summary"
    )
    print(
        "--------------------------------------"
    )

    print(
        f"Baseline samples: "
        f"{baseline['sample_id'].nunique():,}"
    )

    print(
        f"Unique subjects: "
        f"{baseline['subject_id'].nunique():,}"
    )

    print("\nSamples by project:")
    print(
        project_counts.to_string(
            index=False
        )
    )

    print("\nSubjects by response:")
    print(
        response_counts.to_string(
            index=False
        )
    )

    print("\nSubjects by sex:")
    print(
        sex_counts.to_string(
            index=False
        )
    )


# -------------------------------------------------------------------
# Form calculation
# -------------------------------------------------------------------

def calculate_form_answer():
    """
    Average B-cell count for melanoma male responders at
    time zero across all treatments and sample types.
    """

    connection = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            AVG(cc.cell_count) AS average_b_cells,
            COUNT(*) AS sample_count
        FROM subjects AS s
        JOIN samples AS sm
            ON s.subject_id = sm.subject_id
        JOIN cell_counts AS cc
            ON sm.sample_id = cc.sample_id
        WHERE
            LOWER(s.condition) = 'melanoma'
            AND LOWER(s.sex) = 'm'
            AND LOWER(s.response) = 'yes'
            AND sm.time_from_treatment_start = 0
            AND cc.cell_type = 'b_cell';
    """

    result = connection.execute(
        query
    ).fetchone()

    connection.close()

    if result is None or result[0] is None:
        raise RuntimeError(
            "Unable to calculate form answer."
        )

    average_b_cells = result[0]
    sample_count = result[1]

    FORM_ANSWER_OUTPUT.write_text(
        f"{average_b_cells:.2f}\n",
        encoding="utf-8",
    )

    print(
        "\nGoogle Form calculation"
    )
    print(
        "-----------------------"
    )

    print(
        f"Matching baseline samples: "
        f"{sample_count:,}"
    )

    print(
        f"Average B-cell count: "
        f"{average_b_cells:.2f}"
    )

    print(
        f"Answer saved to: "
        f"{FORM_ANSWER_OUTPUT}"
    )


def main():
    calculate_cell_frequencies()
    analyze_responder_differences()
    analyze_baseline_samples()
    calculate_form_answer()


if __name__ == "__main__":
    main()