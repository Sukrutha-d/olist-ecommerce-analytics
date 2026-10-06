import sqlite3
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "db" / "olist.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def test_database_exists():
    """The SQLite warehouse should exist."""

    assert DB_PATH.exists()


def test_fact_orders_grain():
    """
    fact_orders must contain exactly one row per
    order_id + order_item_id.
    """

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(
                DISTINCT order_id || '-' || order_item_id
            ) AS unique_items
        FROM fact_orders;
        """,
        conn
    )

    conn.close()

    assert (
        result.loc[0, "total_rows"]
        == result.loc[0, "unique_items"]
    )


def test_fact_orders_customer_keys():
    """Every fact row should have a customer dimension key."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS missing_customers
        FROM fact_orders
        WHERE customer_key IS NULL;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "missing_customers"] == 0


def test_fact_orders_product_keys():
    """Every fact row should have a product dimension key."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS missing_products
        FROM fact_orders
        WHERE product_key IS NULL;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "missing_products"] == 0


def test_fact_orders_seller_keys():
    """Every fact row should have a seller dimension key."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS missing_sellers
        FROM fact_orders
        WHERE seller_key IS NULL;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "missing_sellers"] == 0


def test_revenue_is_non_negative():
    """Product prices must not be negative."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS invalid_rows
        FROM fact_orders
        WHERE price < 0;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "invalid_rows"] == 0


def test_monthly_business_performance_exists():
    """Business-performance output should exist."""

    conn = get_connection()

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'monthly_business_performance';
        """,
        conn
    )

    conn.close()

    assert len(tables) == 1


def test_forecast_exists():
    """Forecast output should exist."""

    conn = get_connection()

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'revenue_forecast';
        """,
        conn
    )

    conn.close()

    assert len(tables) == 1


def test_forecast_values_non_negative():
    """Revenue forecasts should not be negative."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS invalid_rows
        FROM revenue_forecast
        WHERE forecast_revenue < 0;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "invalid_rows"] == 0


def test_forecast_horizon():
    """The final forecast should contain six months."""

    conn = get_connection()

    result = pd.read_sql_query(
        """
        SELECT COUNT(*) AS forecast_months
        FROM revenue_forecast;
        """,
        conn
    )

    conn.close()

    assert result.loc[0, "forecast_months"] == 6


def test_forecast_model_comparison_exists():
    """Forecast model evaluation should exist."""

    conn = get_connection()

    tables = pd.read_sql_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'forecast_model_comparison';
        """,
        conn
    )

    conn.close()

    assert len(tables) == 1
