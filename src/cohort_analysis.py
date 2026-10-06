"""
Cohort retention analysis for Olist customers.

For each customer:
    1. Find their first purchase month.
    2. Find every subsequent purchase month.
    3. Calculate the month index relative to the first purchase.
    4. Build a cohort retention matrix.

Customer identity:
    customer_unique_id
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
    """Create SQLite connection."""
    return sqlite3.connect(DB_PATH)


def load_orders(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Load customer purchase dates."""

    return pd.read_sql_query(
        """
        SELECT
            order_id,
            customer_unique_id,
            order_purchase_timestamp
        FROM order_features
        WHERE customer_unique_id IS NOT NULL
        """,
        connection,
    )


def prepare_cohort_data(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create first-purchase cohort and month index.

    Month index:
        0 = customer's first purchase month
        1 = one month after first purchase
        2 = two months after first purchase
        ...
    """

    orders = orders.copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    orders = orders.dropna(
        subset=[
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    )

    orders["purchase_month"] = (
        orders["order_purchase_timestamp"]
        .dt.to_period("M")
    )

    first_purchase = (
        orders.groupby("customer_unique_id")[
            "purchase_month"
        ]
        .min()
        .rename("cohort_month")
    )

    orders = orders.merge(
        first_purchase,
        on="customer_unique_id",
        how="left",
    )

    orders["month_index"] = (
        (
            orders["purchase_month"].dt.year
            - orders["cohort_month"].dt.year
        )
        * 12
        +
        (
            orders["purchase_month"].dt.month
            - orders["cohort_month"].dt.month
        )
    )

    return orders


def build_retention_matrix(
    cohort_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build cohort retention matrix.

    Each customer is counted once per cohort/month combination
    even if they placed multiple orders in that month.
    """

    customer_months = (
        cohort_data[
            [
                "customer_unique_id",
                "cohort_month",
                "month_index",
            ]
        ]
        .drop_duplicates()
    )

    cohort_counts = (
        customer_months.groupby(
            ["cohort_month", "month_index"]
        )["customer_unique_id"]
        .nunique()
        .reset_index(name="customers")
    )

    cohort_sizes = (
        cohort_counts[
            cohort_counts["month_index"] == 0
        ][
            ["cohort_month", "customers"]
        ]
        .rename(
            columns={
                "customers": "cohort_size"
            }
        )
    )

    retention = cohort_counts.merge(
        cohort_sizes,
        on="cohort_month",
        how="left",
    )

    retention["retention_rate"] = (
        retention["customers"]
        / retention["cohort_size"]
    )

    matrix = retention.pivot(
        index="cohort_month",
        columns="month_index",
        values="retention_rate",
    )

    return matrix

def calculate_repeat_rate(
    cohort_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the percentage of customers with at least
    two distinct orders.

    Repeat behavior is defined using order count rather than
    month index because two orders can occur within the same
    calendar month.
    """

    order_counts = (
        cohort_data.groupby(
            "customer_unique_id"
        )["order_id"]
        .nunique()
    )

    total_customers = len(order_counts)

    repeat_customers = int(
        (order_counts >= 2).sum()
    )

    repeat_rate = pd.DataFrame(
        {
            "customers": [total_customers],
            "repeat_customers": [repeat_customers],
        }
    )

    repeat_rate["repeat_rate"] = (
        repeat_rate["repeat_customers"]
        / repeat_rate["customers"]
    )

    return repeat_rate


def calculate_time_to_second_order(
    cohort_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate time between first and second orders.

    Uses order-level dates, not simply month index, so the result
    preserves the actual number of days between purchases.
    """

    orders = (
        cohort_data[
            [
                "customer_unique_id",
                "order_id",
                "order_purchase_timestamp",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "customer_unique_id",
                "order_purchase_timestamp",
            ]
        )
    )

    orders["order_number"] = (
        orders.groupby("customer_unique_id")
        .cumcount()
        + 1
    )

    second_orders = orders[
        orders["order_number"] == 2
    ].copy()

    first_orders = orders[
        orders["order_number"] == 1
    ][
        [
            "customer_unique_id",
            "order_purchase_timestamp",
        ]
    ].rename(
        columns={
            "order_purchase_timestamp":
                "first_purchase_timestamp"
        }
    )

    second_orders = second_orders.merge(
        first_orders,
        on="customer_unique_id",
        how="left",
    )

    second_orders["days_to_second_order"] = (
        (
            second_orders["order_purchase_timestamp"]
            - second_orders["first_purchase_timestamp"]
        )
        .dt.total_seconds()
        / 86400
    )

    return second_orders[
        [
            "customer_unique_id",
            "days_to_second_order",
        ]
    ]


def save_results(
    connection: sqlite3.Connection,
    cohort_data: pd.DataFrame,
    retention_matrix: pd.DataFrame,
    repeat_rate: pd.DataFrame,
    time_to_second: pd.DataFrame,
) -> None:
    """Save cohort analysis outputs to SQLite."""

    save_data = cohort_data[
        [
            "order_id",
            "customer_unique_id",
            "purchase_month",
            "cohort_month",
            "month_index",
        ]
    ].copy()

    save_data["purchase_month"] = (
        save_data["purchase_month"].astype(str)
    )

    save_data["cohort_month"] = (
        save_data["cohort_month"].astype(str)
    )

    save_data.to_sql(
        "cohort_customer_months",
        connection,
        if_exists="replace",
        index=False,
    )

    # SQLite cannot directly store pandas Period objects.
    # Convert the cohort-month index to strings before saving.
    retention_matrix_to_save = retention_matrix.copy()

    retention_matrix_to_save.index = (
        retention_matrix_to_save.index.astype(str)
    )

    retention_matrix_to_save.index.name = "cohort_month"

    retention_matrix_to_save.to_sql(
        "cohort_retention",
        connection,
        if_exists="replace",
    )

    repeat_rate.to_sql(
        "cohort_repeat_rate",
        connection,
        if_exists="replace",
        index=False,
    )

    time_to_second.to_sql(
        "time_to_second_order",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.commit()


def validate(
    cohort_data: pd.DataFrame,
    retention_matrix: pd.DataFrame,
) -> None:
    """Validate cohort calculations."""

    if cohort_data["month_index"].min() < 0:
        raise ValueError(
            "Negative cohort month index detected."
        )

    if not retention_matrix.empty:
        diagonal = retention_matrix.iloc[:, 0]

        if not (
            (diagonal >= 0.999)
            & (diagonal <= 1.001)
        ).all():
            raise ValueError(
                "Cohort month-0 retention is not 100%."
            )

    logger.info(
        "Cohort validation passed | customer-month rows=%d",
        len(cohort_data),
    )


def main() -> None:
    """Run cohort retention analysis."""

    logger.info("=" * 70)
    logger.info("OLIST COHORT RETENTION ANALYSIS")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        orders = load_orders(connection)

        logger.info(
            "Loaded order records | rows=%d",
            len(orders),
        )

        cohort_data = prepare_cohort_data(
            orders
        )

        retention_matrix = build_retention_matrix(
            cohort_data
        )

        repeat_rate = calculate_repeat_rate(
            cohort_data
        )

        time_to_second = (
            calculate_time_to_second_order(
                cohort_data
            )
        )

        validate(
            cohort_data,
            retention_matrix,
        )

        save_results(
            connection,
            cohort_data,
            retention_matrix,
            repeat_rate,
            time_to_second,
        )

        logger.info(
            "Unique customers | %d",
            cohort_data[
                "customer_unique_id"
            ].nunique(),
        )

        logger.info(
            "Cohorts | %d",
            cohort_data[
                "cohort_month"
            ].nunique(),
        )

        logger.info(
            "Repeat customers | %d",
            int(
                repeat_rate[
                    "repeat_customers"
                ].iloc[0]
            ),
        )

        logger.info(
            "Repeat rate | %.2f%%",
            100
            * repeat_rate[
                "repeat_rate"
            ].iloc[0],
        )

        if not time_to_second.empty:
            logger.info(
                "Customers with second order | %d",
                len(time_to_second),
            )

            logger.info(
                "Median days to second order | %.1f",
                time_to_second[
                    "days_to_second_order"
                ].median(),
            )

            logger.info(
                "Average days to second order | %.1f",
                time_to_second[
                    "days_to_second_order"
                ].mean(),
            )

        logger.info(
            "Cohort retention matrix saved."
        )

        logger.info("=" * 70)
        logger.info(
            "COHORT ANALYSIS COMPLETE"
        )
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
