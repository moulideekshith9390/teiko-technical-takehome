import csv
import sqlite3
from pathlib import Path

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


# -------------------------------------------------------------------
# Part 2
# -------------------------------------------------------------------

def calculate_cell_frequencies():
    """
    Calculate the relative frequency of each immune-cell population
    for every sample using the SQLite database.
    """

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
            ROUND(
                100.0 * cc.cell_count / totals.total_count,
                4
            ) AS percentage
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
                    row["percentage"],
                ]
            )

    print("Part 2: Cell frequency analysis")
    print("--------------------------------")
    print(f"Rows generated: {len(rows):,}")
    print(f"Output: {FREQUENCY_OUTPUT}")


# -------------------------------------------------------------------
# Part 3 helpers
# -------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """
    Apply the Benjamini-Hochberg procedure to control
    the false discovery rate across multiple hypothesis tests.
    """

    p_values = list(p_values)
    number_of_tests = len(p_values)

    indexed = sorted(
        enumerate(p_values),
        key=lambda item: item[1],
    )

    adjusted = [0.0] * number_of_tests
    previous = 1.0

    for rank_from_end, (original_index, p_value) in enumerate(
        reversed(indexed),
        start=1,
    ):
        rank = number_of_tests - rank_from_end + 1

        corrected = min(
            previous,
            p_value * number_of_tests / rank,
            1.0,
        )

        adjusted[original_index] = corrected
        previous = corrected

    return adjusted


def cohens_d(group_a, group_b):
    """
    Calculate Cohen's d using the pooled standard deviation.
    """

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
    Compare immune-cell frequencies between responders and
    non-responders among melanoma subjects treated with miraclib
    using PBMC samples.

    Because subjects have repeated longitudinal samples, samples
    are first aggregated to one mean frequency per subject and
    population before statistical testing.
    """

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run 'python load_data.py' first."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    query = """
        SELECT
            s.subject_id,
            s.response,
            sm.sample_id,
            cc.cell_type,
            100.0 * cc.cell_count / totals.total_count AS percentage
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

    # Normalize response labels.
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

    # Each subject has longitudinal observations.
    # Average those observations before statistical testing so
    # the independent unit is the subject rather than the sample.
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

    cell_types = sorted(
        subject_level["cell_type"].unique()
    )

    for cell_type in cell_types:

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

        # Welch's t-test does not assume equal group variances.
        test = stats.ttest_ind(
            responders,
            non_responders,
            equal_var=False,
            nan_policy="omit",
        )

        effect_size = cohens_d(
            responders,
            non_responders,
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
                "cohens_d": effect_size,
            }
        )

        # Create one boxplot per immune-cell population.
        figure, axis = plt.subplots(figsize=(6, 5))

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

        display_name = cell_type.replace("_", " ").title()

        axis.set_title(
            f"{display_name}: Responders vs Non-responders"
        )
        axis.set_ylabel("Cell frequency (%)")
        axis.set_xlabel("Clinical response")

        figure.tight_layout()

        figure_path = (
            FIGURE_DIR
            / f"{cell_type}_response_boxplot.png"
        )

        figure.savefig(
            figure_path,
            dpi=300,
        )

        plt.close(figure)

    results_df = pd.DataFrame(results)

    # Correct the five simultaneous hypothesis tests.
    results_df["adjusted_p_value"] = (
        benjamini_hochberg(
            results_df["p_value"].tolist()
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

    print("\nPart 3: Responder vs non-responder analysis")
    print("-------------------------------------------")

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
        results_df[
            [
                "population",
                "responder_mean_pct",
                "non_responder_mean_pct",
                "p_value",
                "adjusted_p_value",
                "cohens_d",
                "significant_fdr_0_05",
            ]
        ].to_string(index=False)
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
# Pipeline
# -------------------------------------------------------------------

def main():

    calculate_cell_frequencies()

    analyze_responder_differences()


if __name__ == "__main__":
    main()