"""
Profile the raw Olist datasets.

This module performs exploratory data-quality checks before
any cleaning or transformation is applied.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from config import DB_PATH


TABLES = [
    "raw_orders",
    "raw_order_items",
    "raw_order_payments",
    "raw_order_reviews",
    "raw_customers",
    "raw_sellers",
    "raw_products",
    "raw_geolocation",
    "raw_category_translation",
]


def get_connection() -> sqlite3.Connection:
    """Create a SQLite database connection."""
    return sqlite3.connect(DB_PATH)


def profile_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> None:
    """Display basic information about a table."""

    df = pd.read_sql_query(
        f"SELECT * FROM {table_name} LIMIT 5",
        connection,
    )

    row_count = pd.read_sql_query(
        f"SELECT COUNT(*) AS count FROM {table_name}",
        connection,
    ).iloc[0]["count"]

    print("\n" + "=" * 70)
    print(f"TABLE: {table_name}")
    print("=" * 70)

    print(f"Rows: {row_count}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nSample:")
    print(df.head(3).to_string(index=False))


def profile_orders(connection: sqlite3.Connection) -> None:
    """Profile order statuses and timestamps."""

    print("\n" + "=" * 70)
    print("ORDER ANALYSIS")
    print("=" * 70)

    status = pd.read_sql_query(
        """
        SELECT
            order_status,
            COUNT(*) AS order_count
        FROM raw_orders
        GROUP BY order_status
        ORDER BY order_count DESC
        """,
        connection,
    )

    print("\nOrder status:")
    print(status.to_string(index=False))

    dates = pd.read_sql_query(
        """
        SELECT
            MIN(order_purchase_timestamp) AS earliest_purchase,
            MAX(order_purchase_timestamp) AS latest_purchase,

            MIN(order_delivered_customer_date)
                AS earliest_delivery,

            MAX(order_delivered_customer_date)
                AS latest_delivery
        FROM raw_orders
        """,
        connection,
    )

    print("\nOrder date range:")
    print(dates.to_string(index=False))

    missing_dates = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_orders,

            SUM(
                order_delivered_customer_date IS NULL
            ) AS missing_delivery_date,

            SUM(
                order_estimated_delivery_date IS NULL
            ) AS missing_estimated_date

        FROM raw_orders
        """,
        connection,
    )

    print("\nMissing delivery dates:")
    print(missing_dates.to_string(index=False))


def profile_customers(connection: sqlite3.Connection) -> None:
    """Profile customer identifiers."""

    print("\n" + "=" * 70)
    print("CUSTOMER ANALYSIS")
    print("=" * 70)

    result = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS customer_records,
            COUNT(DISTINCT customer_id) AS unique_customer_ids,
            COUNT(DISTINCT customer_unique_id)
                AS unique_customer_unique_ids
        FROM raw_customers
        """,
        connection,
    )

    print(result.to_string(index=False))

    repeat_customers = pd.read_sql_query(
        """
        SELECT
            customer_unique_id,
            COUNT(*) AS order_records
        FROM raw_customers
        GROUP BY customer_unique_id
        HAVING COUNT(*) > 1
        ORDER BY order_records DESC
        LIMIT 10
        """,
        connection,
    )

    print("\nExamples of customer_unique_id values with multiple records:")
    print(repeat_customers.to_string(index=False))


def profile_reviews(connection: sqlite3.Connection) -> None:
    """Profile review scores and duplicate order reviews."""

    print("\n" + "=" * 70)
    print("REVIEW ANALYSIS")
    print("=" * 70)

    scores = pd.read_sql_query(
        """
        SELECT
            review_score,
            COUNT(*) AS review_count
        FROM raw_order_reviews
        GROUP BY review_score
        ORDER BY review_score
        """,
        connection,
    )

    print("\nReview score distribution:")
    print(scores.to_string(index=False))

    duplicates = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS duplicated_orders
        FROM (
            SELECT order_id
            FROM raw_order_reviews
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
        """,
        connection,
    )

    print("\nOrders with multiple review records:")
    print(duplicates.to_string(index=False))


def profile_payments(connection: sqlite3.Connection) -> None:
    """Profile payment methods and values."""

    print("\n" + "=" * 70)
    print("PAYMENT ANALYSIS")
    print("=" * 70)

    payment_types = pd.read_sql_query(
        """
        SELECT
            payment_type,
            COUNT(*) AS payment_count,
            ROUND(AVG(payment_value), 2) AS avg_payment_value
        FROM raw_order_payments
        GROUP BY payment_type
        ORDER BY payment_count DESC
        """,
        connection,
    )

    print("\nPayment types:")
    print(payment_types.to_string(index=False))

    negative_payments = pd.read_sql_query(
        """
        SELECT COUNT(*) AS negative_payment_values
        FROM raw_order_payments
        WHERE payment_value < 0
        """,
        connection,
    )

    print("\nNegative payment values:")
    print(negative_payments.to_string(index=False))


def profile_products(connection: sqlite3.Connection) -> None:
    """Profile products and missing product attributes."""

    print("\n" + "=" * 70)
    print("PRODUCT ANALYSIS")
    print("=" * 70)

    result = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS products,

            SUM(
                product_category_name IS NULL
            ) AS missing_category,

            SUM(
                product_weight_g IS NULL
            ) AS missing_weight,

            SUM(
                product_length_cm IS NULL
            ) AS missing_length,

            SUM(
                product_height_cm IS NULL
            ) AS missing_height,

            SUM(
                product_width_cm IS NULL
            ) AS missing_width

        FROM raw_products
        """,
        connection,
    )

    print(result.to_string(index=False))


def profile_order_items(connection: sqlite3.Connection) -> None:
    """Profile order-item prices and freight values."""

    print("\n" + "=" * 70)
    print("ORDER ITEM ANALYSIS")
    print("=" * 70)

    result = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS item_count,

            MIN(price) AS min_price,
            MAX(price) AS max_price,
            AVG(price) AS avg_price,

            MIN(freight_value) AS min_freight,
            MAX(freight_value) AS max_freight,
            AVG(freight_value) AS avg_freight

        FROM raw_order_items
        """,
        connection,
    )

    print(result.to_string(index=False))

    negative = pd.read_sql_query(
        """
        SELECT
            SUM(price < 0) AS negative_prices,
            SUM(freight_value < 0) AS negative_freight
        FROM raw_order_items
        """,
        connection,
    )

    print("\nNegative values:")
    print(negative.to_string(index=False))


def main() -> None:
    """Run the complete raw-data profiling process."""

    print("=" * 70)
    print("OLIST RAW DATA PROFILING")
    print("=" * 70)

    with get_connection() as connection:

        print("\n\nTABLE OVERVIEW")

        for table in TABLES:
            profile_table(connection, table)

        profile_orders(connection)
        profile_customers(connection)
        profile_reviews(connection)
        profile_payments(connection)
        profile_products(connection)
        profile_order_items(connection)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
