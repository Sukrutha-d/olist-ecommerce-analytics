"""
Build the Olist analytical star schema.

Fact grain:
    One row per order item.

Important:
    Payments are aggregated to one row per order before joining
    to the item-level fact table to prevent row multiplication.
"""

from __future__ import annotations
import logging
import sqlite3
from pathlib import Path
import pandas as pd
from config import DB_PATH


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Database helpers
# -------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key enforcement."""
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def read_table(
    connection: sqlite3.Connection,
    table_name: str,
) -> pd.DataFrame:
    """Read a SQLite table into a DataFrame."""
    return pd.read_sql_query(
        f"SELECT * FROM {table_name}",
        connection,
    )


# -------------------------------------------------------------------
# Dimension builders
# -------------------------------------------------------------------

def build_dim_customer(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build the customer dimension.

    customer_unique_id is the real customer identity used by Olist
    for retention/RFM analysis.
    """

    customers = read_table(connection, "clean_customers")

    customers = (
        customers[
            [
                "customer_unique_id",
                "customer_city",
                "customer_state",
            ]
        ]
        .drop_duplicates(subset=["customer_unique_id"])
        .sort_values("customer_unique_id")
        .reset_index(drop=True)
    )

    customers = customers.rename(
        columns={
            "customer_city": "city",
            "customer_state": "state",
        }
    )

    customers.insert(
        0,
        "customer_key",
        range(1, len(customers) + 1),
    )

    customers.to_sql(
        "dim_customer",
        connection,
        if_exists="append",
        index=False,
    )

    return customers


def build_dim_product(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Build the product dimension."""

    products = read_table(connection, "clean_products")

    products = products[
        [
            "product_id",
            "category",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ]
    ].copy()

    products = products.drop_duplicates(
        subset=["product_id"]
    )

    products = products.sort_values(
        "product_id"
    ).reset_index(drop=True)

    products = products.rename(
        columns={
            "product_weight_g": "weight_g",
            "product_length_cm": "length_cm",
            "product_height_cm": "height_cm",
            "product_width_cm": "width_cm",
        }
    )

    products.insert(
        0,
        "product_key",
        range(1, len(products) + 1),
    )

    products.to_sql(
        "dim_product",
        connection,
        if_exists="append",
        index=False,
    )

    return products


def build_dim_seller(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Build the seller dimension."""

    sellers = read_table(connection, "clean_sellers")

    sellers = sellers[
        [
            "seller_id",
            "seller_city",
            "seller_state",
        ]
    ].drop_duplicates(
        subset=["seller_id"]
    )

    sellers = sellers.sort_values(
        "seller_id"
    ).reset_index(drop=True)

    sellers = sellers.rename(
        columns={
            "seller_city": "city",
            "seller_state": "state",
        }
    )

    sellers.insert(
        0,
        "seller_key",
        range(1, len(sellers) + 1),
    )

    sellers.to_sql(
        "dim_seller",
        connection,
        if_exists="append",
        index=False,
    )

    return sellers


def build_dim_date(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build the calendar dimension from order purchase dates.
    """

    orders = read_table(
        connection,
        "clean_orders",
    )

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce",
    )

    dates = (
        orders["order_purchase_timestamp"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    dim_date = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "year": dates.dt.year,
            "month": dates.dt.month,
            "quarter": dates.dt.quarter,
            "day_of_week": dates.dt.dayofweek,
            "is_weekend": (
                dates.dt.dayofweek >= 5
            ).astype(int),
        }
    )

    dim_date.insert(
        0,
        "date_key",
        dates.dt.strftime("%Y%m%d").astype(int),
    )

    dim_date.to_sql(
        "dim_date",
        connection,
        if_exists="append",
        index=False,
    )

    return dim_date


# -------------------------------------------------------------------
# Fact builder
# -------------------------------------------------------------------

def build_fact_orders(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Build the order-item fact table.

    Grain:
        One row per order item.

    Payments are aggregated by order before joining.
    Reviews are already reduced to one row per order.
    """

    logger.info("Reading cleaned source tables...")

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
        "dim_customer",
    )
    
    customer_map = read_table(
	connection,
	"clean_customers",
    )

    products = read_table(
        connection,
        "dim_product",
    )

    sellers = read_table(
        connection,
        "dim_seller",
    )

    dates = read_table(
        connection,
        "dim_date",
    )

    # ---------------------------------------------------------------
    # Parse timestamps
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Aggregate payments to ORDER grain
    #
    # This prevents payment rows from multiplying item rows.
    # ---------------------------------------------------------------

    payment_by_order = (
        payments
        .groupby("order_id", as_index=False)
        .agg(
            payment_value=(
                "payment_value",
                "sum",
            )
        )
    )

    # ---------------------------------------------------------------
    # Keep one review per order
    # ---------------------------------------------------------------

    review_by_order = (
        reviews[
            [
                "order_id",
                "review_score",
            ]
        ]
        .drop_duplicates(
            subset=["order_id"]
        )
    )

    # ---------------------------------------------------------------
    # Calculate delivery metrics at ORDER grain
    # ---------------------------------------------------------------

    delivery = orders[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ].copy()

    delivery["delivery_days"] = (
        (
            delivery["order_delivered_customer_date"]
            - delivery["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 86400
    )

    delivery["estimated_delivery_days"] = (
        (
            delivery["order_estimated_delivery_date"]
            - delivery["order_purchase_timestamp"]
        ).dt.total_seconds()
        / 86400
    )

    delivery["delay_days"] = (
        delivery["order_delivered_customer_date"]
        - delivery["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    delivery["is_late"] = (
        delivery["delay_days"] > 0
    ).astype("Int64")

    # Orders that have not been delivered cannot be classified
    # as late/on-time for delivered-order analysis.
    delivery.loc[
        delivery["order_delivered_customer_date"].isna(),
        "is_late",
    ] = pd.NA

    # ---------------------------------------------------------------
    # Build order-item fact
    # ---------------------------------------------------------------

    fact = order_items[
        [
            "order_id",
            "order_item_id",
            "product_id",
            "seller_id",
            "price",
            "freight_value",
        ]
    ].copy()

    # ---------------------------------------------------------------
    # Attach customer and purchase date
    # ---------------------------------------------------------------

    order_customer = orders[
        [
            "order_id",
            "customer_id",
            "order_purchase_timestamp",
        ]
    ].copy()

    fact = fact.merge(
        order_customer,
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach surrogate dimension keys
    # ---------------------------------------------------------------
    # Map order-level customer_id to the real customer identity.
    # Olist has two customer identifiers:
    #  customer_id   = order-level customer-record
    #  customer_unique_id = actual customer identity
    #
    # Multiple customer_id values can represent the same
    # customer_unique_id ,so we first resolve the mapping
    # through clean_customers.

    fact = fact.merge(
        customer_map[
            [
                "customer_id",
                "customer_unique_id",
            ]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    # convert the real customer identity into the warehouse
    # surrogate key
    fact = fact.merge(
	customers[
	    [
		"customer_key",
		"customer_unique_id",
	    ]
	],
	on="customer_unique_id",
	how="left",
	validate="many_to_one",
    )

    fact = fact.merge(
        products[
            [
                "product_key",
                "product_id",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    fact = fact.merge(
        sellers[
            [
                "seller_key",
                "seller_id",
            ]
        ],
        on="seller_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach date key
    # ---------------------------------------------------------------

    fact["purchase_date"] = (
        fact["order_purchase_timestamp"]
        .dt.strftime("%Y-%m-%d")
    )

    fact = fact.merge(
        dates[
            [
                "date_key",
                "date",
            ]
        ],
        left_on="purchase_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach aggregated payment
    # ---------------------------------------------------------------

    fact = fact.merge(
        payment_by_order,
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach review
    # ---------------------------------------------------------------

    fact = fact.merge(
        review_by_order,
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Attach delivery metrics
    # ---------------------------------------------------------------

    fact = fact.merge(
        delivery[
            [
                "order_id",
                "delivery_days",
                "estimated_delivery_days",
                "delay_days",
                "is_late",
            ]
        ],
        on="order_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------------
    # Select final warehouse columns
    # ---------------------------------------------------------------

    fact = fact[
        [
            "order_id",
            "order_item_id",
            "customer_key",
            "product_key",
            "seller_key",
            "date_key",
            "price",
            "freight_value",
            "payment_value",
            "review_score",
            "delivery_days",
            "estimated_delivery_days",
            "delay_days",
            "is_late",
        ]
    ].copy()

    return fact


# -------------------------------------------------------------------
# Validation
# -------------------------------------------------------------------

def validate_warehouse(
    connection: sqlite3.Connection,
    fact: pd.DataFrame,
) -> None:
    """Run basic warehouse integrity checks."""

    logger.info("Running warehouse validation...")

    # Fact grain uniqueness
    duplicate_fact_keys = fact.duplicated(
        subset=["order_id", "order_item_id"]
    ).sum()

    if duplicate_fact_keys != 0:
        raise ValueError(
            f"Fact grain violation: "
            f"{duplicate_fact_keys} duplicate rows."
        )

    # Expected item count
    clean_item_count = pd.read_sql_query(
        """
        SELECT COUNT(*) AS count
        FROM clean_order_items
        """,
        connection,
    ).iloc[0]["count"]

    if len(fact) != clean_item_count:
        raise ValueError(
            "Fact row count does not match "
            "clean_order_items row count: "
            f"{len(fact)} != {clean_item_count}"
        )

    # Foreign-key null checks
    foreign_key_columns = [
        "customer_key",
        "product_key",
        "seller_key",
        "date_key",
    ]

    for column in foreign_key_columns:
        missing = fact[column].isna().sum()

        if missing != 0:
            raise ValueError(
                f"{column} contains {missing} missing keys."
            )

    logger.info(
        "Fact grain validated | rows=%d",
        len(fact),
    )

    logger.info(
        "Foreign-key completeness validated."
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    """Build and validate the analytical warehouse."""

    logger.info("=" * 70)
    logger.info("BUILDING OLIST DATA WAREHOUSE")
    logger.info("=" * 70)

    connection = get_connection()

    try:
        # Clear previous warehouse data.
        connection.execute("DELETE FROM fact_orders")
        connection.execute("DELETE FROM dim_customer")
        connection.execute("DELETE FROM dim_product")
        connection.execute("DELETE FROM dim_seller")
        connection.execute("DELETE FROM dim_date")

        connection.commit()

        logger.info("Building dimensions...")

        dim_customer = build_dim_customer(
            connection
        )

        dim_product = build_dim_product(
            connection
        )

        dim_seller = build_dim_seller(
            connection
        )

        dim_date = build_dim_date(
            connection
        )

        logger.info(
            "dim_customer | rows=%d",
            len(dim_customer),
        )

        logger.info(
            "dim_product  | rows=%d",
            len(dim_product),
        )

        logger.info(
            "dim_seller   | rows=%d",
            len(dim_seller),
        )

        logger.info(
            "dim_date     | rows=%d",
            len(dim_date),
        )

        logger.info("Building fact_orders...")

        fact = build_fact_orders(
            connection
        )

        fact.to_sql(
            "fact_orders",
            connection,
            if_exists="append",
            index=False,
        )

        connection.commit()

        validate_warehouse(
            connection,
            fact,
        )

        logger.info(
            "fact_orders  | rows=%d",
            len(fact),
        )

        logger.info("=" * 70)
        logger.info("WAREHOUSE BUILD COMPLETE")
        logger.info("=" * 70)

    finally:
        connection.close()


if __name__ == "__main__":
    main()
