"""
Load the raw Olist CSV files into SQLite.

The raw tables are intentionally kept separate from the cleaned
and analytical tables so that the original source data remains
available for reproducibility and auditing.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_DIR, DB_PATH, RAW_DATA_DIR

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

DATASETS = {
    "olist_orders_dataset.csv": "raw_orders",
    "olist_order_items_dataset.csv": "raw_order_items",
    "olist_order_payments_dataset.csv": "raw_order_payments",
    "olist_order_reviews_dataset.csv": "raw_order_reviews",
    "olist_customers_dataset.csv": "raw_customers",
    "olist_sellers_dataset.csv": "raw_sellers",
    "olist_products_dataset.csv": "raw_products",
    "olist_geolocation_dataset.csv": "raw_geolocation",
    "product_category_name_translation.csv": "raw_category_translation",
}


# ---------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------

def validate_input_files() -> None:
    """Verify that all expected CSV files exist."""

    missing_files = []

    for filename in DATASETS:
        filepath = RAW_DATA_DIR / filename

        if not filepath.exists():
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            "The following required CSV files are missing:\n"
            + "\n".join(f"  - {filename}" for filename in missing_files)
        )


def load_csv_to_sqlite(
    connection: sqlite3.Connection,
    filename: str,
    table_name: str,
    chunksize: int = 50_000,
) -> int:
    """
    Load one CSV file into a SQLite table.

    Parameters
    ----------
    connection:
        Active SQLite connection.

    filename:
        Name of the CSV file in data/raw/.

    table_name:
        SQLite table name.

    chunksize:
        Number of rows loaded into memory at a time.

    Returns
    -------
    int
        Total number of rows loaded.
    """

    filepath = RAW_DATA_DIR / filename

    logger.info("Loading %s -> %s", filename, table_name)

    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(filepath, chunksize=chunksize):

        # Replace the existing table only for the first chunk.
        # Subsequent chunks are appended.
        if_exists = "replace" if first_chunk else "append"

        chunk.to_sql(
            table_name,
            connection,
            if_exists=if_exists,
            index=False,
        )

        total_rows += len(chunk)
        first_chunk = False

    logger.info(
        "Loaded %-35s | rows: %d",
        table_name,
        total_rows,
    )

    return total_rows


def load_all_data() -> dict[str, int]:
    """
    Load all raw CSV files into SQLite.

    Returns
    -------
    dict[str, int]
        Mapping of SQLite table names to row counts.
    """

    validate_input_files()

    DB_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Database path: %s", DB_PATH)

    row_counts: dict[str, int] = {}

    with sqlite3.connect(DB_PATH) as connection:

        # Improve SQLite performance during the bulk load.
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")

        for filename, table_name in DATASETS.items():

            row_counts[table_name] = load_csv_to_sqlite(
                connection=connection,
                filename=filename,
                table_name=table_name,
            )

    logger.info("All datasets loaded successfully.")

    return row_counts


def main() -> None:
    """Run the raw data ingestion pipeline."""

    logger.info("=" * 70)
    logger.info("OLIST RAW DATA INGESTION")
    logger.info("=" * 70)

    row_counts = load_all_data()

    logger.info("-" * 70)
    logger.info("LOAD SUMMARY")
    logger.info("-" * 70)

    for table_name, count in row_counts.items():
        logger.info("%-30s : %d rows", table_name, count)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()

