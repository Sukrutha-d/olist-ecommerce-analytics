"""
Delivery performance and customer-outcome analysis.

Questions:
    1. Are delayed deliveries associated with lower review scores?
    2. Is delay correlated with review score?
    3. Does a late first order relate to repeat purchasing?
    4. Which customer states have the highest late-delivery rates?

Statistical tests:
    - Spearman rank correlation
    - Mann-Whitney U test
    - Chi-square test

Effect size:
    - Rank-biserial correlation for Mann-Whitney U
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd
from scipy.stats import (
    chi2_contingency,
    mannwhitneyu,
    spearmanr,
)

from config import DB_PATH


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Create SQLite connection."""
    return sqlite3.connect(DB_PATH)


def load_orders(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Load delivery and customer-outcome data
    from order_features.
    """

    return pd.read_sql_query(
        """
        SELECT
            order_id,
            customer_unique_id,
            order_purchase_timestamp,
            order_delivered_customer_date,
            order_estimated_delivery_date,
            delivery_days,
            estimated_delivery_days,
            delay_days,
            is_late,
            delay_bucket,
            review_score
        FROM order_features
        """,
        connection,
    )


def prepare_delivery_data(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare valid delivered orders for delivery analysis.

    Delivery analysis requires:
        - a customer delivery date
        - a review score
    """

    data = orders.copy()

    data["order_purchase_timestamp"] = pd.to_datetime(
        data["order_purchase_timestamp"],
        errors="coerce",
    )

    # Delivery analysis requires an actual delivery date.
    data = data[
        data["order_delivered_customer_date"].notna()
    ].copy()

    # Review analysis requires a review score.
    data = data[
        data["review_score"].notna()
    ].copy()

    return data


def review_by_delay_bucket(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate review performance by delivery-delay bucket.
    """

    return (
        data.groupby("delay_bucket")
        .agg(
            orders=("order_id", "count"),
            avg_review=("review_score", "mean"),
            median_review=("review_score", "median"),
            avg_delay_days=("delay_days", "mean"),
        )
        .reset_index()
    )


def test_delay_review_relationship(
    data: pd.DataFrame,
) -> dict[str, float]:
    """
    Test the relationship between delivery delay and review score.

    Spearman:
        Measures monotonic association between delay and review.

    Mann-Whitney U:
        Compares review-score distributions between late
        and on-time orders.

    Rank-biserial correlation:
        Measures the effect size of the Mann-Whitney comparison.
    """

    correlation_data = data[
        [
            "delay_days",
            "review_score",
        ]
    ].dropna()

    # ---------------------------------------------------------
    # Spearman correlation
    # ---------------------------------------------------------

    spearman_rho, spearman_p = spearmanr(
        correlation_data["delay_days"],
        correlation_data["review_score"],
    )

    # ---------------------------------------------------------
    # Split review scores into late and on-time groups
    # ---------------------------------------------------------

    late_reviews = data.loc[
        data["is_late"] == 1,
        "review_score",
    ]

    ontime_reviews = data.loc[
        data["is_late"] == 0,
        "review_score",
    ]

    # ---------------------------------------------------------
    # Mann-Whitney U test
    # ---------------------------------------------------------

    u_stat, mannwhitney_p = mannwhitneyu(
        late_reviews,
        ontime_reviews,
        alternative="two-sided",
    )

    # ---------------------------------------------------------
    # Rank-biserial correlation
    #
    # Negative value is expected here because late orders
    # generally have lower review scores.
    # ---------------------------------------------------------

    n_late = len(late_reviews)
    n_ontime = len(ontime_reviews)

    rank_biserial = (
        (2 * u_stat)
        / (n_late * n_ontime)
    ) - 1

    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "mannwhitney_u": u_stat,
        "mannwhitney_p": mannwhitney_p,
        "rank_biserial": rank_biserial,
        "late_mean_review": late_reviews.mean(),
        "ontime_mean_review": ontime_reviews.mean(),
        "late_median_review": late_reviews.median(),
        "ontime_median_review": ontime_reviews.median(),
    }


def calculate_first_order_repeat_rate(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine whether the customer's first order was late
    and whether the customer subsequently placed another order.
    """

    orders = data[
        [
            "customer_unique_id",
            "order_id",
            "order_purchase_timestamp",
            "is_late",
        ]
    ].copy()

    orders = orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    )

    orders["order_number"] = (
        orders.groupby("customer_unique_id")
        .cumcount()
        + 1
    )

    # First order for every customer.
    first_orders = orders[
        orders["order_number"] == 1
    ].copy()

    # Total distinct orders per customer.
    order_counts = (
        orders.groupby("customer_unique_id")[
            "order_id"
        ]
        .nunique()
        .rename("total_orders")
    )

    first_orders = first_orders.merge(
        order_counts,
        on="customer_unique_id",
        how="left",
    )

    # Customer is repeat if they have at least 2 orders.
    first_orders["repeat_customer"] = (
        first_orders["total_orders"] >= 2
    ).astype(int)

    summary = (
        first_orders.groupby("is_late")
        .agg(
            customers=("customer_unique_id", "count"),
            repeat_customers=(
                "repeat_customer",
                "sum",
            ),
            repeat_rate=(
                "repeat_customer",
                "mean",
            ),
        )
        .reset_index()
    )

    return summary


def test_late_first_order_repeat(
    data: pd.DataFrame,
) -> dict[str, float]:
    """
    Perform chi-square test for late first order
    versus repeat purchasing.
    """

    orders = data[
        [
            "customer_unique_id",
            "order_id",
            "order_purchase_timestamp",
            "is_late",
        ]
    ].copy()

    orders = orders.sort_values(
        [
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    )

    orders["order_number"] = (
        orders.groupby("customer_unique_id")
        .cumcount()
        + 1
    )

    first_orders = orders[
        orders["order_number"] == 1
    ].copy()

    order_counts = (
        orders.groupby("customer_unique_id")[
            "order_id"
        ]
        .nunique()
    )

    first_orders["repeat_customer"] = (
        first_orders["customer_unique_id"]
        .map(order_counts)
        >= 2
    ).astype(int)

    # Create 2 x 2 contingency table:
    #
    #              Not Repeat   Repeat
    # On-time
    # Late
    #
    contingency = pd.crosstab(
        first_orders["is_late"],
        first_orders["repeat_customer"],
    )

    chi2, p_value, degrees, expected = (
        chi2_contingency(contingency)
    )

    return {
        "chi2": chi2,
        "p_value": p_value,
        "degrees_of_freedom": degrees,
        "late_repeat_rate": first_orders.loc[
            first_orders["is_late"] == 1,
            "repeat_customer",
        ].mean(),
        "ontime_repeat_rate": first_orders.loc[
            first_orders["is_late"] == 0,
            "repeat_customer",
        ].mean(),
    }


def state_delivery_performance(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Calculate delivery KPIs by customer state.

    Only states with at least 100 delivered orders are included
    so that very small samples do not dominate the ranking.

    avg_late_delay_days:
        Average delay among orders that were actually late.

    This is intentionally different from the overall average
    signed delay, which can be negative when many orders arrive
    earlier than estimated.
    """

    return pd.read_sql_query(
        """
        SELECT
            c.state,

            COUNT(DISTINCT o.order_id)
                AS orders,

            ROUND(
                AVG(o.delivery_days),
                2
            ) AS avg_delivery_days,

            ROUND(
                AVG(
                    CASE
                        WHEN o.delay_days > 0
                        THEN o.delay_days
                    END
                ),
                2
            ) AS avg_late_delay_days,

            ROUND(
                AVG(o.is_late) * 100,
                2
            ) AS late_rate_pct,

            ROUND(
                AVG(o.review_score),
                2
            ) AS avg_review_score

        FROM order_features o

        JOIN dim_customer c
            ON o.customer_unique_id =
               c.customer_unique_id

        WHERE
            o.order_delivered_customer_date IS NOT NULL

        GROUP BY
            c.state

        HAVING
            COUNT(DISTINCT o.order_id) >= 100

        ORDER BY
            late_rate_pct DESC
        """,
        connection,
    )


def main() -> None:
    """Run delivery and customer-outcome analysis."""

    logger.info("=" * 70)
    logger.info(
        "OLIST DELIVERY PERFORMANCE ANALYSIS"
    )
    logger.info("=" * 70)

    connection = get_connection()

    try:

        # =====================================================
        # 1. Load data
        # =====================================================

        orders = load_orders(
            connection
        )

        logger.info(
            "Loaded orders | rows=%d",
            len(orders),
        )

        # =====================================================
        # 2. Prepare delivery data
        # =====================================================

        data = prepare_delivery_data(
            orders
        )

        logger.info(
            "Valid delivered orders with reviews | rows=%d",
            len(data),
        )

        # =====================================================
        # 3. Review score by delay bucket
        # =====================================================

        bucket_summary = (
            review_by_delay_bucket(
                data
            )
        )

        logger.info(
            "Review performance by delay bucket:"
        )

        for _, row in bucket_summary.iterrows():

            logger.info(
                "  %-20s | orders=%d | "
                "avg_review=%.2f | avg_delay=%.2f",
                row["delay_bucket"],
                row["orders"],
                row["avg_review"],
                row["avg_delay_days"],
            )

        # =====================================================
        # 4. Delay vs review statistical analysis
        # =====================================================

        relationship = (
            test_delay_review_relationship(
                data
            )
        )

        logger.info(
            "Spearman delay-review | "
            "rho=%.4f | p=%.6g",
            relationship["spearman_rho"],
            relationship["spearman_p"],
        )

        logger.info(
            "Late orders | mean review=%.3f | median=%.1f",
            relationship["late_mean_review"],
            relationship["late_median_review"],
        )

        logger.info(
            "On-time orders | mean review=%.3f | median=%.1f",
            relationship["ontime_mean_review"],
            relationship["ontime_median_review"],
        )

        logger.info(
            "Mann-Whitney U | "
            "U=%.2f | p=%.6g",
            relationship["mannwhitney_u"],
            relationship["mannwhitney_p"],
        )

        logger.info(
            "Rank-biserial effect size | "
            "r=%.4f",
            relationship["rank_biserial"],
        )

        # =====================================================
        # 5. First-order delay vs repeat purchase
        # =====================================================

        repeat_summary = (
            calculate_first_order_repeat_rate(
                data
            )
        )

        logger.info(
            "First-order late vs repeat summary:"
        )

        for _, row in repeat_summary.iterrows():

            label = (
                "Late"
                if row["is_late"] == 1
                else "On-time"
            )

            logger.info(
                "  %-8s | customers=%d | "
                "repeat=%d | rate=%.2f%%",
                label,
                row["customers"],
                row["repeat_customers"],
                row["repeat_rate"] * 100,
            )

        # =====================================================
        # 6. Chi-square test
        # =====================================================

        repeat_test = (
            test_late_first_order_repeat(
                data
            )
        )

        logger.info(
            "Chi-square late first order vs repeat | "
            "chi2=%.4f | p=%.6g",
            repeat_test["chi2"],
            repeat_test["p_value"],
        )

        logger.info(
            "Late-first-order repeat rate=%.2f%%",
            repeat_test["late_repeat_rate"] * 100,
        )

        logger.info(
            "On-time-first-order repeat rate=%.2f%%",
            repeat_test["ontime_repeat_rate"] * 100,
        )

        # =====================================================
        # 7. State-level delivery performance
        # =====================================================

        state_summary = (
            state_delivery_performance(
                connection
            )
        )

        # Save state results.
        state_summary.to_sql(
            "state_delivery_performance",
            connection,
            if_exists="replace",
            index=False,
        )

        # Save delay-bucket results.
        bucket_summary.to_sql(
            "delivery_review_buckets",
            connection,
            if_exists="replace",
            index=False,
        )

        connection.commit()

        # =====================================================
        # 8. Display worst states
        # =====================================================

        logger.info(
            "Worst states by late-delivery rate:"
        )

        for _, row in state_summary.head(10).iterrows():

            logger.info(
                "  %s | orders=%d | "
                "late=%.2f%% | "
                "avg_late_delay=%.2f | "
                "review=%.2f",
                row["state"],
                row["orders"],
                row["late_rate_pct"],
                row["avg_late_delay_days"],
                row["avg_review_score"],
            )

        logger.info("=" * 70)
        logger.info(
            "DELIVERY ANALYSIS COMPLETE"
        )
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
