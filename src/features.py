"""
Feature engineering for Olist analytics.

Creates:
    1. order_features  - one row per order
    2. customer_features - one row per real customer

The customer identity used here is customer_unique_id.
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
    """Return a SQLite connection with foreign keys enabled."""

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    """Read a SQLite table into a pandas DataFrame."""

    return pd.read_sql_query(
        f"SELECT * FROM {table_name}",
        connection,
    )


# -------------------------------------------------------------------
# Delivery features
# -------------------------------------------------------------------

def create_delivery_features(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate delivery-related features at order grain.
    """

    orders = orders.copy()

    datetime_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for column in datetime_columns:
        orders[column] = pd.to_datetime(
            orders[column],
            errors="coerce",
        )

    # Actual delivery duration.
    orders["delivery_days"] = (
        (
            orders["order_delivered_customer_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 86400
    )

    # Expected delivery duration.
    orders["estimated_delivery_days"] = (
        (
            orders["order_estimated_delivery_date"]
            - orders["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 86400
    )

    # Positive delay means delivery happened after the estimate.
    orders["delay_days"] = (
        (
            orders["order_delivered_customer_date"]
            - orders["order_estimated_delivery_date"]
        ).dt.total_seconds()
        / 86400
    )

    # Only delivered orders can be classified as late/on-time.
    orders["is_late"] = pd.Series(
        pd.NA,
        index=orders.index,
        dtype="Int64",
    )

    delivered_mask = (
        orders["order_delivered_customer_date"].notna()
    )

    orders.loc[
        delivered_mask,
        "is_late",
    ] = (
        orders.loc[
            delivered_mask,
            "delay_days",
        ] > 0
    ).astype(int)

    # ---------------------------------------------------------------
    # Business-friendly delay buckets
    # ---------------------------------------------------------------

    orders["delay_bucket"] = "Not delivered"

    delivered = delivered_mask

    orders.loc[
        delivered & (orders["delay_days"] <= 0),
        "delay_bucket",
    ] = "Early / On Time"

    orders.loc[
        delivered
        & (orders["delay_days"] > 0)
        & (orders["delay_days"] <= 3),
        "delay_bucket",
    ] = "1-3 days late"

    orders.loc[
        delivered
        & (orders["delay_days"] > 3)
        & (orders["delay_days"] <= 7),
        "delay_bucket",
    ] = "4-7 days late"

    orders.loc[
        delivered
        & (orders["delay_days"] > 7)
        & (orders["delay_days"] <= 14),
        "delay_bucket",
    ] = "8-14 days late"

    orders.loc[
        delivered & (orders["delay_days"] > 14),
        "delay_bucket",
    ] = "15+ days late"

    return orders


# -------------------------------------------------------------------
# Order-level features
# -------------------------------------------------------------------

def build_order_features(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build one analytical row per order.
    """

    logger.info("Building order-level features...")

    orders = read_table(
        connection,
        "clean_orders",
    )

    order_items = read_table(
        connection,
        "clean_order_items",
    )

    payments = read_table(
        connection,
        "clean_payments",
    )

    reviews = read_table(
        connection,
        "clean_reviews",
    )

    customers = read_table(
        connection,
        "clean_customers",
    )

    # ---------------------------------------------------------------
    # Delivery metrics
    # ---------------------------------------------------------------

    orders = create_delivery_features(
        orders
    )

    # ---------------------------------------------------------------
    # Map to real customer identity
    # ---------------------------------------------------------------

    orders = orders.merge(
        customers[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Order item aggregates
    # ---------------------------------------------------------------

    item_features = (
        order_items
        .groupby("order_id", as_index=False)
        .agg(
            order_total=(
                "price",
                "sum",
            ),
            total_items=(
                "order_item_id",
                "count",
            ),
            total_freight=(
                "freight_value",
                "sum",
            ),
        )
    )

    # ---------------------------------------------------------------
    # Payment aggregates
    # ---------------------------------------------------------------

    payment_features = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            payment_value=(
                "payment_value",
                "sum",
            ),
            payment_type=(
                "payment_type",
                lambda x: "|".join(
                    sorted(
                        x.dropna()
                        .astype(str)
                        .unique()
                    )
                ),
            ),
            installments=(
                "payment_installments",
                "max",
            ),
        )
    )

    # ---------------------------------------------------------------
    # Review
    # ---------------------------------------------------------------

    review_features = reviews[
        [
            "order_id",
            "review_score",
        ]
    ].drop_duplicates(
        subset=["order_id"]
    )

    # ---------------------------------------------------------------
    # Combine features
    # ---------------------------------------------------------------

    order_features = orders[
        [
            "order_id",
            "customer_unique_id",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
            "delivery_days",
            "estimated_delivery_days",
            "delay_days",
            "is_late",
            "delay_bucket",
        ]
    ].copy()

    order_features = order_features.merge(
        item_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    order_features = order_features.merge(
        payment_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    order_features = order_features.merge(
        review_features,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------------
    # Freight ratio
    # ---------------------------------------------------------------

    order_features["freight_ratio"] = (
        order_features["total_freight"]
        / order_features["order_total"]
    )

    # Avoid division-by-zero cases.
    order_features.loc[
        order_features["order_total"] <= 0,
        "freight_ratio",
    ] = pd.NA

    # ---------------------------------------------------------------
    # Order date
    # ---------------------------------------------------------------

    order_features["order_date"] = (
        pd.to_datetime(
            order_features[
                "order_purchase_timestamp"
            ],
            errors="coerce",
        ).dt.date
    )

    return order_features


# -------------------------------------------------------------------
# Customer features
# -------------------------------------------------------------------

def build_customer_features(
    order_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per real customer.

    Frequency is the number of distinct orders.
    Monetary is total product value across orders.
    """

    logger.info("Building customer-level features...")

    orders = order_features.copy()

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    customer_features = (
        orders
        .groupby(
            "customer_unique_id",
            as_index=False,
        )
        .agg(
            first_purchase_date=(
                "order_purchase_timestamp",
                "min",
            ),
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
            total_freight=(
                "total_freight",
                "sum",
            ),
            average_order_value=(
                "order_total",
                "mean",
            ),
        )
    )

    customer_features["repeat_customer"] = (
        customer_features["frequency"] > 1
    ).astype(int)

    return customer_features


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_features(
    connection: sqlite3.Connection,
    order_features: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> None:
    """Validate feature table grains and key values."""

    logger.info("Running feature validation...")

    # ---------------------------------------------------------------
    # Order grain
    # ---------------------------------------------------------------

    duplicate_orders = (
        order_features["order_id"]
        .duplicated()
        .sum()
    )

    if duplicate_orders != 0:
        raise ValueError(
            f"Order grain violation: "
            f"{duplicate_orders} duplicate order rows."
        )

    source_orders = pd.read_sql_query(
        """
        SELECT COUNT(*) AS count
        FROM clean_orders
        """,
        connection,
    ).iloc[0]["count"]

    if len(order_features) != source_orders:
        raise ValueError(
            "Order feature count does not match "
            f"clean_orders: {len(order_features)} != "
            f"{source_orders}"
        )

    # ---------------------------------------------------------------
    # Customer grain
    # ---------------------------------------------------------------

    duplicate_customers = (
        customer_features["customer_unique_id"]
        .duplicated()
        .sum()
    )

    if duplicate_customers != 0:
        raise ValueError(
            "Customer feature grain contains duplicates."
        )

    # ---------------------------------------------------------------
    # Monetary sanity
    # ---------------------------------------------------------------

    if (
        order_features["order_total"]
        .dropna()
        < 0
    ).any():
        raise ValueError(
            "Negative order totals detected."
        )

    # ---------------------------------------------------------------
    # Delivery sanity
    # ---------------------------------------------------------------

    negative_delivery = (
        order_features["delivery_days"]
        .dropna()
        < 0
    ).sum()

    if negative_delivery != 0:
        raise ValueError(
            f"{negative_delivery} orders have "
            "negative delivery duration."
        )

    logger.info(
        "Order feature grain validated | rows=%d",
        len(order_features),
    )

    logger.info(
        "Customer feature grain validated | rows=%d",
        len(customer_features),
    )


# -------------------------------------------------------------------
# Save tables
# -------------------------------------------------------------------

def save_features(
    connection: sqlite3.Connection,
    order_features: pd.DataFrame,
    customer_features: pd.DataFrame,
) -> None:
    """Save analytical feature tables to SQLite."""

    order_features.to_sql(
        "order_features",
        connection,
        if_exists="replace",
        index=False,
    )

    customer_features.to_sql(
        "customer_features",
        connection,
        if_exists="replace",
        index=False,
    )

    # Indexes support the upcoming analytical queries.
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_order_features_customer
        ON order_features(customer_unique_id)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_order_features_date
        ON order_features(order_date)
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_customer_features_customer
        ON customer_features(customer_unique_id)
        """
    )

    connection.commit()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Build, validate, and save analytical features."""

    logger.info("=" * 70)
    logger.info("BUILDING OLIST FEATURES")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        order_features = build_order_features(
            connection
        )

        customer_features = build_customer_features(
            order_features
        )

        validate_features(
            connection,
            order_features,
            customer_features,
        )

        save_features(
            connection,
            order_features,
            customer_features,
        )

        logger.info(
            "order_features   | rows=%d",
            len(order_features),
        )

        logger.info(
            "customer_features | rows=%d",
            len(customer_features),
        )

        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
