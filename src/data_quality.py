"""
Data quality validation for the Olist dataset.

This module checks:
1. Primary-key uniqueness
2. Duplicate order items
3. Referential integrity
4. Date consistency
5. Invalid numeric values
6. Missing values
7. Payment consistency
8. Review duplication
"""

import logging
import sqlite3
from typing import Dict

import pandas as pd

from config import DB_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Create a connection to the Olist SQLite database."""
    return sqlite3.connect(DB_PATH)


def check_primary_keys(conn: sqlite3.Connection) -> None:
    """Check uniqueness of important identifier columns."""

    logger.info("=" * 70)
    logger.info("PRIMARY KEY / UNIQUENESS CHECKS")
    logger.info("=" * 70)

    checks = {
        "raw_orders": "order_id",
        "raw_customers": "customer_id",
        "raw_products": "product_id",
        "raw_sellers": "seller_id",
    }

    for table, column in checks.items():

        query = f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT {column}) AS unique_values
            FROM {table}
        """

        result = pd.read_sql_query(query, conn).iloc[0]

        total = int(result["total_rows"])
        unique = int(result["unique_values"])

        duplicates = total - unique

        logger.info(
            "%s.%s -> rows=%d, unique=%d, duplicates=%d",
            table,
            column,
            total,
            unique,
            duplicates,
        )


def check_order_item_duplicates(conn: sqlite3.Connection) -> None:
    """Check whether order item identifiers are duplicated."""

    logger.info("=" * 70)
    logger.info("ORDER ITEM DUPLICATE CHECK")
    logger.info("=" * 70)

    query = """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT order_id || '-' || order_item_id) AS unique_items
        FROM raw_order_items
    """

    result = pd.read_sql_query(query, conn).iloc[0]

    total = int(result["total_rows"])
    unique = int(result["unique_items"])

    logger.info(
        "Total rows=%d | Unique (order_id, order_item_id)=%d | Duplicates=%d",
        total,
        unique,
        total - unique,
    )


def check_foreign_keys(conn: sqlite3.Connection) -> None:
    """Check important foreign-key relationships."""

    logger.info("=" * 70)
    logger.info("REFERENTIAL INTEGRITY CHECKS")
    logger.info("=" * 70)

    checks = {
        "order_items -> orders": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_items oi
            LEFT JOIN raw_orders o
                ON oi.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
        "order_items -> products": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_items oi
            LEFT JOIN raw_products p
                ON oi.product_id = p.product_id
            WHERE p.product_id IS NULL
        """,
        "order_items -> sellers": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_items oi
            LEFT JOIN raw_sellers s
                ON oi.seller_id = s.seller_id
            WHERE s.seller_id IS NULL
        """,
        "orders -> customers": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_orders o
            LEFT JOIN raw_customers c
                ON o.customer_id = c.customer_id
            WHERE c.customer_id IS NULL
        """,
        "payments -> orders": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_payments p
            LEFT JOIN raw_orders o
                ON p.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
        "reviews -> orders": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_reviews r
            LEFT JOIN raw_orders o
                ON r.order_id = o.order_id
            WHERE o.order_id IS NULL
        """,
    }

    for relationship, query in checks.items():

        result = pd.read_sql_query(query, conn).iloc[0]
        invalid = int(result["invalid_count"])

        logger.info(
            "%s -> invalid records=%d",
            relationship,
            invalid,
        )


def check_date_consistency(conn: sqlite3.Connection) -> None:
    """Check whether order timestamps follow logical chronological order."""

    logger.info("=" * 70)
    logger.info("DATE CONSISTENCY CHECKS")
    logger.info("=" * 70)

    checks = {
        "approval before purchase": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_orders
            WHERE order_approved_at IS NOT NULL
              AND datetime(order_approved_at) < datetime(order_purchase_timestamp)
        """,
        "carrier delivery before purchase": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_orders
            WHERE order_delivered_carrier_date IS NOT NULL
              AND datetime(order_delivered_carrier_date)
                  < datetime(order_purchase_timestamp)
        """,
        "customer delivery before purchase": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_orders
            WHERE order_delivered_customer_date IS NOT NULL
              AND datetime(order_delivered_customer_date)
                  < datetime(order_purchase_timestamp)
        """,
        "customer delivery before carrier delivery": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_orders
            WHERE order_delivered_customer_date IS NOT NULL
              AND order_delivered_carrier_date IS NOT NULL
              AND datetime(order_delivered_customer_date)
                  < datetime(order_delivered_carrier_date)
        """,
    }

    for check_name, query in checks.items():

        result = pd.read_sql_query(query, conn).iloc[0]
        invalid = int(result["invalid_count"])

        logger.info(
            "%s -> invalid records=%d",
            check_name,
            invalid,
        )


def check_numeric_ranges(conn: sqlite3.Connection) -> None:
    """Check for impossible numeric values."""

    logger.info("=" * 70)
    logger.info("NUMERIC RANGE CHECKS")
    logger.info("=" * 70)

    checks = {
        "negative order item price": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_items
            WHERE price < 0
        """,
        "negative freight value": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_items
            WHERE freight_value < 0
        """,
        "negative payment value": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_payments
            WHERE payment_value < 0
        """,
        "invalid review score": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_reviews
            WHERE review_score NOT BETWEEN 1 AND 5
        """,
        "invalid payment installments": """
            SELECT COUNT(*) AS invalid_count
            FROM raw_order_payments
            WHERE payment_installments < 0
        """,
    }

    for check_name, query in checks.items():

        result = pd.read_sql_query(query, conn).iloc[0]
        invalid = int(result["invalid_count"])

        logger.info(
            "%s -> invalid records=%d",
            check_name,
            invalid,
        )


def check_missing_values(conn: sqlite3.Connection) -> None:
    """Check important missing values."""

    logger.info("=" * 70)
    logger.info("IMPORTANT MISSING VALUE CHECKS")
    logger.info("=" * 70)

    checks = {
        "orders.purchase_timestamp": """
            SELECT COUNT(*) AS missing_count
            FROM raw_orders
            WHERE order_purchase_timestamp IS NULL
        """,
        "orders.estimated_delivery": """
            SELECT COUNT(*) AS missing_count
            FROM raw_orders
            WHERE order_estimated_delivery_date IS NULL
        """,
        "customers.customer_unique_id": """
            SELECT COUNT(*) AS missing_count
            FROM raw_customers
            WHERE customer_unique_id IS NULL
        """,
        "products.category": """
            SELECT COUNT(*) AS missing_count
            FROM raw_products
            WHERE product_category_name IS NULL
        """,
        "products.weight": """
            SELECT COUNT(*) AS missing_count
            FROM raw_products
            WHERE product_weight_g IS NULL
        """,
    }

    for field, query in checks.items():

        result = pd.read_sql_query(query, conn).iloc[0]
        missing = int(result["missing_count"])

        logger.info(
            "%s -> missing=%d",
            field,
            missing,
        )


def check_payment_consistency(conn: sqlite3.Connection) -> None:
    """Check payment sequence uniqueness within orders."""

    logger.info("=" * 70)
    logger.info("PAYMENT CONSISTENCY CHECK")
    logger.info("=" * 70)

    query = """
        SELECT COUNT(*) AS duplicate_count
        FROM (
            SELECT order_id, payment_sequential
            FROM raw_order_payments
            GROUP BY order_id, payment_sequential
            HAVING COUNT(*) > 1
        )
    """

    result = pd.read_sql_query(query, conn).iloc[0]
    duplicates = int(result["duplicate_count"])

    logger.info(
        "Duplicate (order_id, payment_sequential) groups=%d",
        duplicates,
    )


def check_review_duplicates(conn: sqlite3.Connection) -> None:
    """Check multiple reviews for the same order."""

    logger.info("=" * 70)
    logger.info("REVIEW DUPLICATE CHECK")
    logger.info("=" * 70)

    query = """
        SELECT
            COUNT(*) AS duplicate_order_groups
        FROM (
            SELECT order_id
            FROM raw_order_reviews
            GROUP BY order_id
            HAVING COUNT(*) > 1
        )
    """

    result = pd.read_sql_query(query, conn).iloc[0]
    duplicates = int(result["duplicate_order_groups"])

    logger.info(
        "Orders with multiple review records=%d",
        duplicates,
    )


def main() -> None:
    """Run all data quality checks."""

    logger.info("=" * 70)
    logger.info("OLIST DATA QUALITY VALIDATION")
    logger.info("=" * 70)

    conn = get_connection()

    try:
        check_primary_keys(conn)
        check_order_item_duplicates(conn)
        check_foreign_keys(conn)
        check_date_consistency(conn)
        check_numeric_ranges(conn)
        check_missing_values(conn)
        check_payment_consistency(conn)
        check_review_duplicates(conn)

        logger.info("=" * 70)
        logger.info("DATA QUALITY VALIDATION COMPLETE")
        logger.info("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
