import csv
import sqlite3
from pathlib import Path


# -------------------------------------------------------------------
# File locations
# -------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "cell-count.csv"
DB_PATH = ROOT_DIR / "cell_counts.db"


CELL_TYPES = [
    "b_cell",
    "cd8_t_cell",
    "cd4_t_cell",
    "nk_cell",
    "monocyte",
]


# -------------------------------------------------------------------
# Database schema
# -------------------------------------------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE projects (
    project_id TEXT PRIMARY KEY
);

CREATE TABLE subjects (
    subject_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    age INTEGER NOT NULL,
    sex TEXT NOT NULL,
    treatment TEXT NOT NULL,
    response TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(project_id)
);

CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    sample_type TEXT NOT NULL,
    time_from_treatment_start INTEGER NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE cell_counts (
    sample_id TEXT NOT NULL,
    cell_type TEXT NOT NULL,
    cell_count INTEGER NOT NULL CHECK (cell_count >= 0),
    PRIMARY KEY (sample_id, cell_type),
    FOREIGN KEY (sample_id) REFERENCES samples(sample_id)
);
"""


def create_database():
    """Create a fresh SQLite database."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {CSV_PATH}"
        )

    # Rebuild the database from scratch each time.
    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    connection.executescript(SCHEMA)

    return connection


def load_csv(connection):
    """Load cell-count.csv into the normalized database."""

    projects = set()
    subjects = {}

    sample_rows = []
    cell_count_rows = []

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "project",
            "subject",
            "condition",
            "age",
            "sex",
            "treatment",
            "response",
            "sample",
            "sample_type",
            "time_from_treatment_start",
            *CELL_TYPES,
        }

        missing_columns = required_columns - set(reader.fieldnames or [])

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            project_id = row["project"]
            subject_id = row["subject"]
            sample_id = row["sample"]

            projects.add(project_id)

            # Subject-level attributes repeat across longitudinal samples,
            # so store each subject only once.
            if subject_id not in subjects:
                response = row["response"].strip() or None

                subjects[subject_id] = (
                    subject_id,
                    project_id,
                    row["condition"],
                    int(row["age"]),
                    row["sex"],
                    row["treatment"],
                    response,
                )

            sample_rows.append(
                (
                    sample_id,
                    subject_id,
                    row["sample_type"],
                    int(row["time_from_treatment_start"]),
                )
            )

            # Convert the five wide cell-count columns into long format.
            for cell_type in CELL_TYPES:
                cell_count_rows.append(
                    (
                        sample_id,
                        cell_type,
                        int(row[cell_type]),
                    )
                )

    with connection:
        connection.executemany(
            """
            INSERT INTO projects (project_id)
            VALUES (?)
            """,
            [(project,) for project in sorted(projects)],
        )

        connection.executemany(
            """
            INSERT INTO subjects (
                subject_id,
                project_id,
                condition,
                age,
                sex,
                treatment,
                response
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            list(subjects.values()),
        )

        connection.executemany(
            """
            INSERT INTO samples (
                sample_id,
                subject_id,
                sample_type,
                time_from_treatment_start
            )
            VALUES (?, ?, ?, ?)
            """,
            sample_rows,
        )

        connection.executemany(
            """
            INSERT INTO cell_counts (
                sample_id,
                cell_type,
                cell_count
            )
            VALUES (?, ?, ?)
            """,
            cell_count_rows,
        )


def validate_database(connection):
    """Print simple checks to verify the database load."""

    tables = [
        "projects",
        "subjects",
        "samples",
        "cell_counts",
    ]

    print("\nDatabase validation")
    print("-------------------")

    for table in tables:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count:,} rows")

    foreign_key_errors = connection.execute(
        "PRAGMA foreign_key_check;"
    ).fetchall()

    if foreign_key_errors:
        raise RuntimeError(
            f"Foreign key validation failed: {foreign_key_errors}"
        )

    integrity_result = connection.execute(
        "PRAGMA integrity_check;"
    ).fetchone()[0]

    if integrity_result != "ok":
        raise RuntimeError(
            f"SQLite integrity check failed: {integrity_result}"
        )

    print("Foreign keys: OK")
    print("Database integrity: OK")


def main():
    connection = create_database()

    try:
        load_csv(connection)
        validate_database(connection)
    finally:
        connection.close()

    print(f"\nDatabase created successfully:")
    print(DB_PATH)


if __name__ == "__main__":
    main()