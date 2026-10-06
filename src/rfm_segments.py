"""
Rule-based RFM customer segmentation.

Uses the RFM table created by src.rfm.

Important:
    Frequency is highly skewed in the Olist dataset, with the vast
    majority of customers making only one purchase. Therefore,
    frequency uses explicit behavioral thresholds instead of
    quantile-based scoring.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection."""
    return sqlite3.connect(DB_PATH)


def load_rfm(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load the customer-level RFM table."""
    return pd.read_sql_query(
        """
        SELECT
            customer_unique_id,
            recency,
            frequency,
            monetary,
            last_purchase_date
        FROM rfm
        """,
        connection,
    )


def assign_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assign business-rule-based R, F and M scores.

    Recency:
        <= 30 days  -> 5
        <= 90 days  -> 4
        <= 180 days -> 3
        <= 365 days -> 2
        > 365 days  -> 1

    Frequency:
        1 order     -> 1
        2 orders    -> 3
        3 orders    -> 4
        >=4 orders  -> 5

    Monetary:
        Quintile-based scoring from 1 to 5.
    """
    result = rfm.copy()

    # More recent customers receive higher scores.
    result["r_score"] = pd.cut(
        result["recency"],
        bins=[-float("inf"), 30, 90, 180, 365, float("inf")],
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    # Explicit behavioral thresholds because frequency is highly skewed.
    result["f_score"] = result["frequency"].apply(
        lambda x: (
            1 if x == 1
            else 3 if x == 2
            else 4 if x == 3
            else 5
        )
    )

    # Monetary quintiles.
    result["m_score"] = pd.qcut(
        result["monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    result["rfm_score"] = (
        result["r_score"]
        + result["f_score"]
        + result["m_score"]
    )

    return result


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign interpretable customer segments."""
    result = rfm.copy()

    def segment(row: pd.Series) -> str:
        r = row["r_score"]
        f = row["f_score"]
        m = row["m_score"]

        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        if r >= 4 and f >= 3:
            return "Loyal Customers"

        if r >= 4 and f <= 2:
            return "New / Promising"

        if r <= 2 and f >= 3 and m >= 4:
            return "At Risk High Value"

        if r <= 2 and f >= 3:
            return "At Risk"

        if r <= 2 and m >= 4:
            return "Lost High Value"

        if r <= 2:
            return "Lost Customers"

        if m >= 4:
            return "Big Spenders"

        return "Needs Attention"

    result["segment"] = result.apply(segment, axis=1)

    return result


def validate_segments(rfm: pd.DataFrame) -> None:
    """Validate segmentation output."""
    required_columns = {
        "customer_unique_id",
        "recency",
        "frequency",
        "monetary",
        "r_score",
        "f_score",
        "m_score",
        "rfm_score",
        "segment",
    }

    missing = required_columns - set(rfm.columns)

    if missing:
        raise ValueError(
            f"Missing expected columns: {sorted(missing)}"
        )

    if rfm["customer_unique_id"].duplicated().any():
        raise ValueError("Duplicate customers found.")

    if rfm["segment"].isna().any():
        raise ValueError("Missing customer segments.")

    logger.info(
        "RFM segmentation validation passed | customers=%d",
        len(rfm),
    )


def save_segments(
    connection: sqlite3.Connection,
    rfm: pd.DataFrame,
) -> None:
    """Save segmented RFM data to SQLite."""
    rfm.to_sql(
        "rfm_segments",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_rfm_segments_customer
        ON rfm_segments(customer_unique_id)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_rfm_segments_segment
        ON rfm_segments(segment)
        """
    )

    connection.commit()


def main() -> None:
    """Run RFM scoring and segmentation."""
    logger.info("=" * 70)
    logger.info("OLIST RFM SEGMENTATION")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        rfm = load_rfm(connection)

        logger.info(
            "Loaded RFM records | rows=%d",
            len(rfm),
        )

        rfm = assign_scores(rfm)
        rfm = assign_segments(rfm)

        validate_segments(rfm)
        save_segments(connection, rfm)

        logger.info("Segment distribution:")

        distribution = (
            rfm["segment"]
            .value_counts()
            .sort_values(ascending=False)
        )

        for segment_name, count in distribution.items():
            percentage = 100 * count / len(rfm)

            logger.info(
                "  %-22s | %6d | %6.2f%%",
                segment_name,
                count,
                percentage,
            )

        logger.info("=" * 70)
        logger.info("RFM SEGMENTATION COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
