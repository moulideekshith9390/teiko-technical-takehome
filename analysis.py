import csv
import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "cell_counts.db"
OUTPUT_DIR = ROOT_DIR / "outputs"
FREQUENCY_OUTPUT = OUTPUT_DIR / "cell_frequencies.csv"


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

    print("\nFirst 10 rows:")
    for row in rows[:10]:
        print(
            row["sample"],
            row["total_count"],
            row["population"],
            row["count"],
            row["percentage"],
        )


def main():
    calculate_cell_frequencies()


if __name__ == "__main__":
    main()