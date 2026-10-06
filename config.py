
"""Central configuration for the Olist analytics project."""

from pathlib import Path

# Project root: directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "olist.db"
SCHEMA_PATH = DB_DIR / "schema.sql"

# Analysis and reporting
SQL_DIR = PROJECT_ROOT / "sql"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TESTS_DIR = PROJECT_ROOT / "tests"

# Reproducibility
RANDOM_STATE = 42

# Business-analysis configuration
REPEAT_PURCHASE_WINDOW_DAYS = 180
LOW_REVIEW_THRESHOLD = 2

# Supported delay categories
DELAY_BUCKETS = [
    "On time",
    "1-3 days late",
    "4-7 days late",
    ">7 days late",
]

def ensure_directories() -> None:
    """Create the directories required by the project."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        DB_DIR,
        SQL_DIR,
        NOTEBOOKS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        DASHBOARD_DIR,
        DASHBOARD_DIR / "pages",
        TESTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print(f"Project root: {PROJECT_ROOT}")
    print("Project directories created successfully.")
