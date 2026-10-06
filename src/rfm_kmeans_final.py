"""
Final K-Means RFM customer clustering.

Uses K=4 selected after comparing:
    - elbow/inertia
    - silhouette score
    - business interpretability

RFM features are:
    1. log1p transformed
    2. standardized
    3. clustered using K-Means
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_CLUSTERS = 4


def get_connection() -> sqlite3.Connection:
    """Create SQLite connection."""
    return sqlite3.connect(DB_PATH)


def load_rfm(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load customer-level RFM metrics."""
    return pd.read_sql_query(
        """
        SELECT
            customer_unique_id,
            recency,
            frequency,
            monetary
        FROM rfm
        """,
        connection,
    )


def prepare_features(
    rfm: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Log-transform and standardize RFM variables."""

    features = rfm[
        ["recency", "frequency", "monetary"]
    ].copy()

    log_features = np.log1p(features)

    scaler = StandardScaler()

    scaled = scaler.fit_transform(log_features)

    scaled_features = pd.DataFrame(
        scaled,
        columns=[
            "recency",
            "frequency",
            "monetary",
        ],
        index=rfm.index,
    )

    return scaled_features, scaler


def train_kmeans(
    scaled_features: pd.DataFrame,
) -> KMeans:
    """Train final K-Means model."""
    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    model.fit(scaled_features)

    return model


def create_cluster_profile(
    rfm: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the business characteristics of each cluster.
    """

    profile = (
        rfm.groupby("cluster")
        .agg(
            customers=("customer_unique_id", "count"),
            avg_recency=("recency", "mean"),
            median_recency=("recency", "median"),
            avg_frequency=("frequency", "mean"),
            median_frequency=("frequency", "median"),
            avg_monetary=("monetary", "mean"),
            median_monetary=("monetary", "median"),
        )
        .reset_index()
    )

    profile["customer_percentage"] = (
        100
        * profile["customers"]
        / len(rfm)
    )

    return profile


def save_results(
    connection: sqlite3.Connection,
    rfm: pd.DataFrame,
    profile: pd.DataFrame,
) -> None:
    """Save clusters and cluster profiles."""

    rfm.to_sql(
        "rfm_kmeans",
        connection,
        if_exists="replace",
        index=False,
    )

    profile.to_sql(
        "rfm_kmeans_profile",
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


def main() -> None:
    """Run final RFM K-Means clustering."""

    logger.info("=" * 70)
    logger.info("OLIST FINAL RFM K-MEANS")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        rfm = load_rfm(connection)

        logger.info(
            "Loaded RFM records | rows=%d",
            len(rfm),
        )

        scaled_features, _ = prepare_features(rfm)

        logger.info(
            "Prepared log1p + standardized RFM features"
        )

        model = train_kmeans(scaled_features)

        rfm["cluster"] = model.labels_

        logger.info(
            "K-Means trained | clusters=%d",
            N_CLUSTERS,
        )

        profile = create_cluster_profile(rfm)

        logger.info("Cluster profile:")

        for _, row in profile.sort_values(
            "cluster"
        ).iterrows():

            logger.info(
                "Cluster %d | customers=%d (%.2f%%) | "
                "avg_recency=%.1f | avg_frequency=%.2f | "
                "avg_monetary=%.2f",
                int(row["cluster"]),
                int(row["customers"]),
                row["customer_percentage"],
                row["avg_recency"],
                row["avg_frequency"],
                row["avg_monetary"],
            )

        save_results(
            connection,
            rfm,
            profile,
        )

        logger.info(
            "K-Means results saved to SQLite"
        )

        logger.info("=" * 70)
        logger.info(
            "FINAL RFM K-MEANS COMPLETE"
        )
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
