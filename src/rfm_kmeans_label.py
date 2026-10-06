"""
Assign business-friendly labels to final RFM K-Means clusters.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


CLUSTER_LABELS = {
    0: "High-Value Inactive",
    1: "Recent One-Time",
    2: "Repeat Customers",
    3: "Low-Value Inactive",
}


def main() -> None:
    """Add business labels to K-Means clusters."""

    logger.info("=" * 70)
    logger.info("LABELING RFM K-MEANS CLUSTERS")
    logger.info("=" * 70)

    connection = sqlite3.connect(DB_PATH)

    try:
        rfm = pd.read_sql_query(
            """
            SELECT
                customer_unique_id,
                recency,
                frequency,
                monetary,
                cluster
            FROM rfm_kmeans
            """,
            connection,
        )

        unknown_clusters = set(rfm["cluster"]) - set(CLUSTER_LABELS)

        if unknown_clusters:
            raise ValueError(
                f"Unknown clusters found: {unknown_clusters}"
            )

        rfm["cluster_label"] = rfm["cluster"].map(
            CLUSTER_LABELS
        )

        rfm.to_sql(
            "rfm_kmeans",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_rfm_kmeans_customer
            ON rfm_kmeans(customer_unique_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_rfm_kmeans_cluster
            ON rfm_kmeans(cluster)
            """
        )

        connection.commit()

        logger.info(
            "Cluster labels saved successfully."
        )

        distribution = (
            rfm["cluster_label"]
            .value_counts()
        )

        for label, count in distribution.items():
            percentage = (
                100 * count / len(rfm)
            )

            logger.info(
                "%-22s | %6d | %6.2f%%",
                label,
                count,
                percentage,
            )

        logger.info("=" * 70)
        logger.info("RFM K-MEANS LABELING COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
