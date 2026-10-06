
"""Create the initial Olist analytics project scaffold."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRECTORIES = [
    "data/raw",
    "data/processed",
    "db",
    "src",
    "sql",
    "notebooks",
    "dashboard/pages",
    "reports/figures",
    "tests",
]

FILES = [
    "README.md",
    "requirements.txt",
    "config.py",
    "db/schema.sql",
    "src/__init__.py",
    "src/load_data.py",
    "src/clean_data.py",
    "src/build_warehouse.py",
    "src/data_quality.py",
    "src/features.py",
    "src/rfm.py",
    "src/cohort.py",
    "src/delivery_analysis.py",
    "src/forecasting.py",
    "src/churn_model.py",
    "src/utils.py",
    "dashboard/app.py",
    "dashboard/components.py",
    "run_pipeline.py",
    "tests/test_data_quality.py",
    "reports/insights_summary.md",
]

def create_scaffold() -> None:
    """Create project directories and missing starter files."""
    for directory in DIRECTORIES:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        path = ROOT / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    print(f"Scaffold created at: {ROOT}")

if __name__ == "__main__":
    create_scaffold()
