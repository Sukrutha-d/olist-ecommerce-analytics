"""
Clean and validate Olist raw data.

Cleaning principles:
- Never modify raw tables.
- Parse timestamp columns to datetime.
- Keep all order statuses in the cleaned orders table.
- Remove impossible delivery timestamps by setting the invalid field to NULL.
- Keep only the latest review for each order.
- Translate product categories to English.
- Fill missing/unmapped categories with "unknown".
- Preserve valid missing product dimensions.
- Validate non-negative monetary values.
- Save cleaned tables to data/processed/ and SQLite.
- Produce a before/after cleaning summary.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

from config import DB_PATH, PROCESSED_DATA_DIR, ensure_directories


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def parse_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert specified columns to pandas datetime.

    Invalid datetime strings become NaT rather than crashing the
    pipeline.
    """
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def save_processed_table(
    df: pd.DataFrame,
    table_name: str,
    conn: sqlite3.Connection,
) -> None:
    """
    Save a cleaned dataframe as both Parquet and SQLite.
    """

    ensure_directories()

    parquet_path = PROCESSED_DATA_DIR / f"{table_name}.parquet"

    df.to_parquet(
        parquet_path,
        index=False,
    )

    df.to_sql(
        f"clean_{table_name}",
        conn,
        if_exists="replace",
        index=False,
    )

    logger.info(
        "Saved %-20s | rows=%8d | parquet=%s",
        table_name,
        len(df),
        parquet_path,
    )


# ---------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------

def clean_orders(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean the orders table.

    Invalid delivery timestamps are set to NULL rather than deleting
    the entire order.
    """

    df = pd.read_sql_query(
        "SELECT * FROM raw_orders",
        conn,
    )

    before_rows = len(df)

    datetime_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    df = parse_datetime_columns(
        df,
        datetime_columns,
    )

    # -------------------------------------------------------------
    # Impossible carrier delivery before purchase
    # -------------------------------------------------------------

    carrier_before_purchase = (
        df["order_delivered_carrier_date"].notna()
        & (
            df["order_delivered_carrier_date"]
            < df["order_purchase_timestamp"]
        )
    )

    carrier_before_purchase_count = int(
        carrier_before_purchase.sum()
    )

    # Remove the impossible timestamp, not the complete order.
    df.loc[
        carrier_before_purchase,
        "order_delivered_carrier_date",
    ] = pd.NaT

    # -------------------------------------------------------------
    # Customer delivery before carrier delivery
    # -------------------------------------------------------------

    customer_before_carrier = (
        df["order_delivered_customer_date"].notna()
        & df["order_delivered_carrier_date"].notna()
        & (
            df["order_delivered_customer_date"]
            < df["order_delivered_carrier_date"]
        )
    )

    customer_before_carrier_count = int(
        customer_before_carrier.sum()
    )

    # The carrier timestamp is the inconsistent field.
    df.loc[
        customer_before_carrier,
        "order_delivered_carrier_date",
    ] = pd.NaT

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "carrier_before_purchase_fixed": carrier_before_purchase_count,
        "customer_before_carrier_fixed": customer_before_carrier_count,
    }

    return df, summary


# ---------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------

def clean_reviews(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean reviews.

    If an order has multiple reviews, retain the latest review based
    on review_answer_timestamp.
    """

    df = pd.read_sql_query(
        "SELECT * FROM raw_order_reviews",
        conn,
    )

    before_rows = len(df)

    datetime_columns = [
        "review_creation_date",
        "review_answer_timestamp",
    ]

    df = parse_datetime_columns(
        df,
        datetime_columns,
    )

    # Sort chronologically so the final record per order is the latest.
    df = df.sort_values(
        by=[
            "order_id",
            "review_answer_timestamp",
            "review_creation_date",
        ],
        na_position="first",
    )

    duplicate_orders = int(
        df.loc[df["order_id"].duplicated(keep=False),"order_id"].nunique()
    )

    # Keep latest review per order.
    df = df.drop_duplicates(
        subset=["order_id"],
        keep="last",
    )

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "review_rows_removed": before_rows - len(df),
        "orders_with_multiple_reviews": duplicate_orders,
    }

    return df, summary


# ---------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------

def clean_products(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean products and translate category names.
    """

    products = pd.read_sql_query(
        "SELECT * FROM raw_products",
        conn,
    )

    translation = pd.read_sql_query(
        "SELECT * FROM raw_category_translation",
        conn,
    )

    before_rows = len(products)

    # -------------------------------------------------------------
    # Category translation
    # -------------------------------------------------------------

    products = products.merge(
        translation,
        on="product_category_name",
        how="left",
    )

    products["product_category_name_english"] = (
        products["product_category_name_english"]
        .fillna("unknown")
        .astype(str)
    )

    # Preserve the original category and expose a clean category.
    products["category"] = (
        products["product_category_name_english"]
    )

    summary = {
        "before_rows": before_rows,
        "after_rows": len(products),
        "missing_original_category": int(
            products["product_category_name"].isna().sum()
        ),
        "unknown_clean_category": int(
            (products["category"] == "unknown").sum()
        ),
    }

    return products, summary


# ---------------------------------------------------------------------
# Order items
# ---------------------------------------------------------------------

def clean_order_items(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean order items.

    No negative prices or freight values were found during profiling,
    but we validate them again here.
    """

    df = pd.read_sql_query(
        "SELECT * FROM raw_order_items",
        conn,
    )

    before_rows = len(df)

    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"],
        errors="coerce",
    )

    invalid_price = df["price"] < 0
    invalid_freight = df["freight_value"] < 0

    invalid_count = int(
        (invalid_price | invalid_freight).sum()
    )

    # Remove only records containing impossible monetary values.
    df = df.loc[
        ~(invalid_price | invalid_freight)
    ].copy()

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "invalid_monetary_rows_removed": invalid_count,
    }

    return df, summary


# ---------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------

def clean_payments(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean payment records.
    """

    df = pd.read_sql_query(
        "SELECT * FROM raw_order_payments",
        conn,
    )

    before_rows = len(df)

    invalid_payment = df["payment_value"] < 0
    invalid_installments = df["payment_installments"] < 0

    invalid_mask = (
        invalid_payment
        | invalid_installments
    )

    invalid_count = int(invalid_mask.sum())

    df = df.loc[
        ~invalid_mask
    ].copy()

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "invalid_payment_rows_removed": invalid_count,
    }

    return df, summary


# ---------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------

def clean_customers(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Clean customer records."""

    df = pd.read_sql_query(
        "SELECT * FROM raw_customers",
        conn,
    )

    before_rows = len(df)

    df = df.drop_duplicates(
        subset=["customer_id"],
        keep="first",
    )

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
    }

    return df, summary


# ---------------------------------------------------------------------
# Sellers
# ---------------------------------------------------------------------

def clean_sellers(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Clean seller records."""

    df = pd.read_sql_query(
        "SELECT * FROM raw_sellers",
        conn,
    )

    before_rows = len(df)

    df = df.drop_duplicates(
        subset=["seller_id"],
        keep="first",
    )

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
    }

    return df, summary


# ---------------------------------------------------------------------
# Geolocation
# ---------------------------------------------------------------------

def clean_geolocation(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean geolocation data.

    We preserve all records because multiple coordinates can legitimately
    exist for the same ZIP-code prefix.
    """

    df = pd.read_sql_query(
        "SELECT * FROM raw_geolocation",
        conn,
    )

    before_rows = len(df)

    # Remove only exact duplicate rows.
    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "exact_duplicates_removed": before_rows - len(df),
    }

    return df, summary


# ---------------------------------------------------------------------
# Category translation
# ---------------------------------------------------------------------

def clean_category_translation(
    conn: sqlite3.Connection,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    """Clean category translation lookup."""

    df = pd.read_sql_query(
        "SELECT * FROM raw_category_translation",
        conn,
    )

    before_rows = len(df)

    df = df.drop_duplicates(
        subset=["product_category_name"],
        keep="first",
    )

    df = df.reset_index(drop=True)

    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
    }

    return df, summary


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def print_cleaning_summary(
    summaries: Dict[str, Dict[str, int]],
) -> None:
    """Print before/after cleaning summary."""

    logger.info("=" * 70)
    logger.info("CLEANING SUMMARY")
    logger.info("=" * 70)

    for table_name, summary in summaries.items():

        before = summary.get("before_rows", 0)
        after = summary.get("after_rows", 0)

        logger.info(
            "%-20s | before=%8d | after=%8d | rows_removed=%8d",
            table_name,
            before,
            after,
            before - after,
        )

        for key, value in summary.items():

            if key in {"before_rows", "after_rows"}:
                continue

            logger.info(
                "    %-40s : %d",
                key,
                value,
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete cleaning pipeline."""

    logger.info("=" * 70)
    logger.info("OLIST DATA CLEANING")
    logger.info("=" * 70)

    ensure_directories()

    conn = get_connection()

    summaries = {}

    try:

        # -------------------------------------------------------------
        # Orders
        # -------------------------------------------------------------

        orders, summary = clean_orders(conn)
        summaries["orders"] = summary

        save_processed_table(
            orders,
            "orders",
            conn,
        )

        # -------------------------------------------------------------
        # Reviews
        # -------------------------------------------------------------

        reviews, summary = clean_reviews(conn)
        summaries["reviews"] = summary

        save_processed_table(
            reviews,
            "reviews",
            conn,
        )

        # -------------------------------------------------------------
        # Products
        # -------------------------------------------------------------

        products, summary = clean_products(conn)
        summaries["products"] = summary

        save_processed_table(
            products,
            "products",
            conn,
        )

        # -------------------------------------------------------------
        # Order items
        # -------------------------------------------------------------

        order_items, summary = clean_order_items(conn)
        summaries["order_items"] = summary

        save_processed_table(
            order_items,
            "order_items",
            conn,
        )

        # -------------------------------------------------------------
        # Payments
        # -------------------------------------------------------------

        payments, summary = clean_payments(conn)
        summaries["payments"] = summary

        save_processed_table(
            payments,
            "payments",
            conn,
        )

        # -------------------------------------------------------------
        # Customers
        # -------------------------------------------------------------

        customers, summary = clean_customers(conn)
        summaries["customers"] = summary

        save_processed_table(
            customers,
            "customers",
            conn,
        )

        # -------------------------------------------------------------
        # Sellers
        # -------------------------------------------------------------

        sellers, summary = clean_sellers(conn)
        summaries["sellers"] = summary

        save_processed_table(
            sellers,
            "sellers",
            conn,
        )

        # -------------------------------------------------------------
        # Geolocation
        # -------------------------------------------------------------

        geolocation, summary = clean_geolocation(conn)
        summaries["geolocation"] = summary

        save_processed_table(
            geolocation,
            "geolocation",
            conn,
        )

        # -------------------------------------------------------------
        # Category translation
        # -------------------------------------------------------------

        translation, summary = clean_category_translation(conn)
        summaries["category_translation"] = summary

        save_processed_table(
            translation,
            "category_translation",
            conn,
        )

        # -------------------------------------------------------------
        # Print final summary
        # -------------------------------------------------------------

        print_cleaning_summary(summaries)

        logger.info("=" * 70)
        logger.info("CLEANING COMPLETE")
        logger.info("=" * 70)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
