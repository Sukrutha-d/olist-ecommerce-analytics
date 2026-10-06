"""
Create a cohort retention heatmap.

The heatmap displays retention percentage by:
    - first purchase cohort
    - months since first purchase

Incomplete future periods are left blank rather than interpreted
as zero retention.
"""

from __future__ import annotations

import logging
import sqlite3

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_retention_matrix(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load the cohort retention matrix."""

    retention = pd.read_sql_query(
        """
        SELECT *
        FROM cohort_retention
        """,
        connection,
    )

    retention = retention.set_index(
        "cohort_month"
    )

    retention.columns = [
        int(column)
        for column in retention.columns
    ]

    return retention


def create_heatmap(
    retention: pd.DataFrame,
) -> None:
    """Create and save the cohort retention heatmap."""

    retention_percent = retention * 100

    plt.figure(
        figsize=(16, 10)
    )

    sns.heatmap(
        retention_percent,
        mask=retention_percent.isna(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={
            "label": "Retention (%)"
        },
    )

    plt.title(
        "Olist Customer Cohort Retention"
    )

    plt.xlabel(
        "Months Since First Purchase"
    )

    plt.ylabel(
        "First Purchase Cohort"
    )

    plt.tight_layout()

    output_path = (
        "reports/cohort_retention_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Heatmap saved to %s",
        output_path,
    )


def main() -> None:
    """Generate cohort retention heatmap."""

    logger.info("=" * 70)
    logger.info(
        "OLIST COHORT RETENTION HEATMAP"
    )
    logger.info("=" * 70)

    connection = sqlite3.connect(DB_PATH)

    try:
        retention = load_retention_matrix(
            connection
        )

        logger.info(
            "Loaded retention matrix | rows=%d | columns=%d",
            retention.shape[0],
            retention.shape[1],
        )

        create_heatmap(
            retention
        )

        logger.info("=" * 70)
        logger.info(
            "COHORT HEATMAP COMPLETE"
        )
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
