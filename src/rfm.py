"""
RFM analysis for Olist customers.

RFM:
    Recency  = days since the customer's last purchase
    Frequency = number of distinct orders
    Monetary = total order value

Customer identity:
    customer_unique_id
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from config import DB_PATH


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Database
# -------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection."""

    return sqlite3.connect(DB_PATH)


# -------------------------------------------------------------------
# RFM calculation
# -------------------------------------------------------------------

def calculate_rfm(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Calculate customer-level RFM metrics.

    Snapshot date:
        Maximum observed purchase date + 1 day.

    This ensures the most recent customer's recency is 1 day
    rather than 0 days.
    """

    orders = pd.read_sql_query(
        """
        SELECT
            order_id,
            customer_unique_id,
            order_purchase_timestamp,
            order_total
        FROM order_features
        """,
        connection,
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    # Only rows with a valid customer and purchase timestamp
    # can contribute to RFM.
    orders = orders[
        orders["customer_unique_id"].notna()
        & orders["order_purchase_timestamp"].notna()
    ].copy()

    # ---------------------------------------------------------------
    # Snapshot date
    # ---------------------------------------------------------------

    snapshot_date = (
        orders["order_purchase_timestamp"].max()
        + pd.Timedelta(days=1)
    )

    logger.info(
        "RFM snapshot date: %s",
        snapshot_date.date(),
    )

    # ---------------------------------------------------------------
    # RFM aggregation
    # ---------------------------------------------------------------

    rfm = (
        orders
        .groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            last_purchase_date=(
                "order_purchase_timestamp",
                "max",
            ),
            frequency=(
                "order_id",
                "nunique",
            ),
            monetary=(
                "order_total",
                "sum",
            ),
        )
    )

    # ---------------------------------------------------------------
    # Recency
    # ---------------------------------------------------------------

    rfm["recency"] = (
        snapshot_date
        - rfm["last_purchase_date"]
    ).dt.days

    # Arrange in standard RFM order.
    rfm = rfm[
        [
            "customer_unique_id",
            "recency",
            "frequency",
            "monetary",
            "last_purchase_date",
        ]
    ]

    return rfm


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_rfm(
    rfm: pd.DataFrame,
) -> None:
    """Validate the RFM table."""

    if rfm["customer_unique_id"].duplicated().any():
        raise ValueError(
            "RFM table contains duplicate customers."
        )

    if (rfm["recency"] < 0).any():
        raise ValueError(
            "Negative recency values detected."
        )

    if (rfm["frequency"] <= 0).any():
        raise ValueError(
            "Invalid frequency values detected."
        )

    if (rfm["monetary"] < 0).any():
        raise ValueError(
            "Negative monetary values detected."
        )

    logger.info(
        "RFM validation passed | customers=%d",
        len(rfm),
    )


# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------

def save_rfm(
    connection: sqlite3.Connection,
    rfm: pd.DataFrame,
) -> None:
    """Save RFM results to SQLite."""

    rfm.to_sql(
        "rfm",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_rfm_customer
        ON rfm(customer_unique_id)
        """
    )

    connection.commit()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Calculate, validate, and save RFM metrics."""

    logger.info("=" * 70)
    logger.info("OLIST RFM ANALYSIS")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        rfm = calculate_rfm(
            connection
        )

        validate_rfm(
            rfm
        )

        save_rfm(
            connection,
            rfm
        )

        logger.info(
            "RFM table saved | rows=%d",
            len(rfm),
        )

        logger.info(
            "Recency  | min=%d | median=%.1f | max=%d",
            rfm["recency"].min(),
            rfm["recency"].median(),
            rfm["recency"].max(),
        )

        logger.info(
            "Frequency | min=%d | median=%.1f | max=%d",
            rfm["frequency"].min(),
            rfm["frequency"].median(),
            rfm["frequency"].max(),
        )

        logger.info(
            "Monetary | min=%.2f | median=%.2f | max=%.2f",
            rfm["monetary"].min(),
            rfm["monetary"].median(),
            rfm["monetary"].max(),
        )

        logger.info("=" * 70)
        logger.info("RFM CALCULATION COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
