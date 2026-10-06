"""
K-Means clustering for Olist RFM customers.

Pipeline:
    RFM metrics
        -> log1p transformation
        -> standardization
        -> K-Means
        -> elbow + silhouette evaluation

Random seed is fixed for reproducibility.
"""

from __future__ import annotations

import logging
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
K_VALUES = range(2, 9)


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection."""
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


def prepare_features(rfm: pd.DataFrame):
    """
    Apply log transformation and standardization.

    log1p is used because RFM variables are highly right-skewed.
    """
    features = rfm[
        ["recency", "frequency", "monetary"]
    ].copy()

    log_features = pd.DataFrame(
        {
            "recency": __import__("numpy").log1p(
                features["recency"]
            ),
            "frequency": __import__("numpy").log1p(
                features["frequency"]
            ),
            "monetary": __import__("numpy").log1p(
                features["monetary"]
            ),
        }
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(log_features)

    scaled_df = pd.DataFrame(
        scaled,
        columns=["recency", "frequency", "monetary"],
        index=rfm.index,
    )

    return scaled_df


def evaluate_k_values(
    scaled_features: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate K-Means using inertia and silhouette score."""
    results = []

    for k in K_VALUES:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        silhouette = silhouette_score(
            scaled_features,
            labels,
        )

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette,
            }
        )

        logger.info(
            "K=%d | inertia=%.2f | silhouette=%.4f",
            k,
            model.inertia_,
            silhouette,
        )

    return pd.DataFrame(results)


def save_evaluation(results: pd.DataFrame) -> None:
    """Save clustering evaluation results and plots."""
    results.to_csv(
        "reports/rfm_kmeans_evaluation.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["k"],
        results["inertia"],
        marker="o",
    )
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("RFM K-Means Elbow Curve")
    plt.xticks(results["k"])
    plt.tight_layout()
    plt.savefig(
        "reports/rfm_kmeans_elbow.png",
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        results["k"],
        results["silhouette"],
        marker="o",
    )
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("RFM K-Means Silhouette Scores")
    plt.xticks(results["k"])
    plt.tight_layout()
    plt.savefig(
        "reports/rfm_kmeans_silhouette.png",
        dpi=150,
    )
    plt.close()


def main() -> None:
    """Run K-Means evaluation."""
    logger.info("=" * 70)
    logger.info("OLIST RFM K-MEANS ANALYSIS")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        rfm = load_rfm(connection)

        logger.info(
            "Loaded RFM records | rows=%d",
            len(rfm),
        )

        scaled_features = prepare_features(rfm)

        logger.info(
            "RFM features transformed using log1p + StandardScaler"
        )

        results = evaluate_k_values(
            scaled_features
        )

        save_evaluation(results)

        logger.info(
            "Evaluation saved to reports/"
        )

        logger.info("=" * 70)
        logger.info("K-MEANS EVALUATION COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
