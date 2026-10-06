"""
Business performance analysis for Olist.

Produces:
    1. Monthly revenue trend
    2. Monthly order/customer metrics
    3. Product-category performance
    4. Pareto analysis
    5. Seller performance
    6. State performance

Revenue definition:
    Revenue = sum of product item prices for delivered orders.
    Freight is reported separately.

Fact table grain:
    fact_orders = one row per order item.

Important:
    payment_value is an order-level value repeated across item rows,
    so it is NOT summed here.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from config import DB_PATH, REPORTS_DIR


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection to the Olist warehouse."""
    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------
# 1. MONTHLY REVENUE
# ---------------------------------------------------------------------

def monthly_revenue(connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Calculate monthly delivered-order revenue.

    Revenue is based on item price.
    Freight is reported separately.
    """

    query = """
        SELECT
            strftime(
                '%Y-%m',
                co.order_purchase_timestamp
            ) AS month,

            COUNT(
                DISTINCT fo.order_id
            ) AS orders,

            COUNT(
                DISTINCT dc.customer_unique_id
            ) AS customers,

            SUM(
                fo.price
            ) AS revenue,

            SUM(
                fo.freight_value
            ) AS freight,

            SUM(
                fo.price + fo.freight_value
            ) AS revenue_plus_freight,

            AVG(
                fo.price
            ) AS avg_item_value

        FROM fact_orders AS fo

        JOIN dim_customer AS dc
            ON fo.customer_key = dc.customer_key

        JOIN clean_orders AS co
            ON fo.order_id = co.order_id

        WHERE co.order_status = 'delivered'

        GROUP BY
            strftime(
                '%Y-%m',
                co.order_purchase_timestamp
            )

        ORDER BY month;
    """

    df = pd.read_sql_query(query, connection)

    if not df.empty:
        df["revenue"] = df["revenue"].round(2)
        df["freight"] = df["freight"].round(2)
        df["revenue_plus_freight"] = df["revenue_plus_freight"].round(2)
        df["avg_item_value"] = df["avg_item_value"].round(2)

    return df


# ---------------------------------------------------------------------
# 2. MONTHLY GROWTH
# ---------------------------------------------------------------------

def add_monthly_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add month-over-month revenue growth."""

    result = df.copy()

    result["revenue_growth_pct"] = (
        result["revenue"]
        .pct_change()
        .mul(100)
        .round(2)
    )

    return result


# ---------------------------------------------------------------------
# 3. CATEGORY PERFORMANCE
# ---------------------------------------------------------------------

def category_performance(
    connection: sqlite3.Connection
) -> pd.DataFrame:
    """
    Calculate revenue and order-item performance by product category.
    """

    query = """
        SELECT
            COALESCE(
                dp.category,
                'unknown'
            ) AS category,

            COUNT(*) AS order_items,

            COUNT(
                DISTINCT fo.order_id
            ) AS orders,

            SUM(
                fo.price
            ) AS revenue,

            SUM(
                fo.freight_value
            ) AS freight,

            AVG(
                fo.price
            ) AS avg_item_value

        FROM fact_orders AS fo

        JOIN dim_product AS dp
            ON fo.product_key = dp.product_key

        JOIN clean_orders AS co
            ON fo.order_id = co.order_id

        WHERE co.order_status = 'delivered'

        GROUP BY
            COALESCE(
                dp.category,
                'unknown'
            )

        ORDER BY revenue DESC;
    """

    df = pd.read_sql_query(query, connection)

    if not df.empty:
        df["revenue"] = df["revenue"].round(2)
        df["freight"] = df["freight"].round(2)
        df["avg_item_value"] = df["avg_item_value"].round(2)

    return df


# ---------------------------------------------------------------------
# 4. PARETO ANALYSIS
# ---------------------------------------------------------------------

def pareto_analysis(
    category_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate cumulative revenue contribution by category.

    Used to identify the categories responsible for approximately
    80% of total revenue.
    """

    df = category_df.copy()

    if df.empty:
        return df

    total_revenue = df["revenue"].sum()

    if total_revenue == 0:
        df["revenue_share_pct"] = 0.0
        df["cumulative_revenue_pct"] = 0.0
        df["pareto_80"] = False
        return df

    df["revenue_share_pct"] = (
        df["revenue"]
        / total_revenue
        * 100
    )

    df["cumulative_revenue_pct"] = (
        df["revenue_share_pct"]
        .cumsum()
    )

    df["revenue_share_pct"] = (
        df["revenue_share_pct"]
        .round(2)
    )

    df["cumulative_revenue_pct"] = (
        df["cumulative_revenue_pct"]
        .round(2)
    )

    df["pareto_80"] = (
        df["cumulative_revenue_pct"] <= 80
    )

    return df


# ---------------------------------------------------------------------
# 5. SELLER PERFORMANCE
# ---------------------------------------------------------------------

def seller_performance(
    connection: sqlite3.Connection
) -> pd.DataFrame:
    """
    Calculate seller-level revenue and order-item performance.
    """

    query = """
        SELECT
            ds.seller_id,
            ds.city,
            ds.state,

            COUNT(*) AS order_items,

            COUNT(
                DISTINCT fo.order_id
            ) AS orders,

            SUM(
                fo.price
            ) AS revenue,

            SUM(
                fo.freight_value
            ) AS freight,

            AVG(
                fo.price
            ) AS avg_item_value

        FROM fact_orders AS fo

        JOIN dim_seller AS ds
            ON fo.seller_key = ds.seller_key

        JOIN clean_orders AS co
            ON fo.order_id = co.order_id

        WHERE co.order_status = 'delivered'

        GROUP BY
            ds.seller_id,
            ds.city,
            ds.state

        ORDER BY revenue DESC;
    """

    df = pd.read_sql_query(query, connection)

    if not df.empty:
        df["revenue"] = df["revenue"].round(2)
        df["freight"] = df["freight"].round(2)
        df["avg_item_value"] = df["avg_item_value"].round(2)

    return df


# ---------------------------------------------------------------------
# 6. STATE PERFORMANCE
# ---------------------------------------------------------------------

def state_performance(
    connection: sqlite3.Connection
) -> pd.DataFrame:
    """
    Calculate delivered-order performance by customer state.
    """

    query = """
        SELECT
            dc.state,

            COUNT(
                DISTINCT fo.order_id
            ) AS orders,

            COUNT(
                DISTINCT dc.customer_unique_id
            ) AS customers,

            SUM(
                fo.price
            ) AS revenue,

            SUM(
                fo.freight_value
            ) AS freight,

            AVG(
                fo.price
            ) AS avg_item_value

        FROM fact_orders AS fo

        JOIN dim_customer AS dc
            ON fo.customer_key = dc.customer_key

        JOIN clean_orders AS co
            ON fo.order_id = co.order_id

        WHERE co.order_status = 'delivered'

        GROUP BY
            dc.state

        ORDER BY revenue DESC;
    """

    df = pd.read_sql_query(query, connection)

    if not df.empty:
        df["revenue"] = df["revenue"].round(2)
        df["freight"] = df["freight"].round(2)
        df["avg_item_value"] = df["avg_item_value"].round(2)

    return df


# ---------------------------------------------------------------------
# 7. TOP CATEGORIES
# ---------------------------------------------------------------------

def top_categories(
    category_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """Return the top N categories by revenue."""

    return (
        category_df
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(n)
        .copy()
    )


# ---------------------------------------------------------------------
# 8. TOP SELLERS
# ---------------------------------------------------------------------

def top_sellers(
    seller_df: pd.DataFrame,
    n: int = 20
) -> pd.DataFrame:
    """Return the top N sellers by revenue."""

    return (
        seller_df
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(n)
        .copy()
    )


# ---------------------------------------------------------------------
# 9. TOP STATES
# ---------------------------------------------------------------------

def top_states(
    state_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """Return the top N customer states by revenue."""

    return (
        state_df
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(n)
        .copy()
    )


# ---------------------------------------------------------------------
# 10. SAVE RESULTS
# ---------------------------------------------------------------------

def save_results(
    connection: sqlite3.Connection,
    monthly: pd.DataFrame,
    categories: pd.DataFrame,
    pareto: pd.DataFrame,
    sellers: pd.DataFrame,
    states: pd.DataFrame
) -> None:
    """Save business-performance outputs to SQLite and CSV files."""

    # SQLite tables
    monthly.to_sql(
        "monthly_business_performance",
        connection,
        if_exists="replace",
        index=False
    )

    categories.to_sql(
        "category_performance",
        connection,
        if_exists="replace",
        index=False
    )

    pareto.to_sql(
        "category_pareto",
        connection,
        if_exists="replace",
        index=False
    )

    sellers.to_sql(
        "seller_performance",
        connection,
        if_exists="replace",
        index=False
    )

    states.to_sql(
        "state_business_performance",
        connection,
        if_exists="replace",
        index=False
    )

    connection.commit()

    # CSV outputs
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    monthly.to_csv(
        REPORTS_DIR / "monthly_business_performance.csv",
        index=False
    )

    categories.to_csv(
        REPORTS_DIR / "category_performance.csv",
        index=False
    )

    pareto.to_csv(
        REPORTS_DIR / "category_pareto.csv",
        index=False
    )

    sellers.to_csv(
        REPORTS_DIR / "seller_performance.csv",
        index=False
    )

    states.to_csv(
        REPORTS_DIR / "state_business_performance.csv",
        index=False
    )

    logger.info(
        "Business-performance results saved to SQLite and reports/"
    )


# ---------------------------------------------------------------------
# 11. PRINT SUMMARY
# ---------------------------------------------------------------------

def print_summary(
    monthly: pd.DataFrame,
    categories: pd.DataFrame,
    pareto: pd.DataFrame,
    sellers: pd.DataFrame,
    states: pd.DataFrame
) -> None:
    """Print key business findings."""

    logger.info("=" * 70)
    logger.info("BUSINESS PERFORMANCE SUMMARY")
    logger.info("=" * 70)

    if monthly.empty:
        logger.warning("Monthly revenue table is empty.")
        return

    # Overall revenue
    total_revenue = monthly["revenue"].sum()
    total_freight = monthly["freight"].sum()
    total_orders = monthly["orders"].sum()

    logger.info(
        "Total delivered revenue: %.2f",
        total_revenue
    )

    logger.info(
        "Total freight: %.2f",
        total_freight
    )

    logger.info(
        "Total delivered orders: %d",
        total_orders
    )

    # Best month
    best_month = monthly.loc[
        monthly["revenue"].idxmax()
    ]

    logger.info(
        "Best revenue month: %s | revenue=%.2f",
        best_month["month"],
        best_month["revenue"]
    )

    # Worst month
    worst_month = monthly.loc[
        monthly["revenue"].idxmin()
    ]

    logger.info(
        "Lowest revenue month: %s | revenue=%.2f",
        worst_month["month"],
        worst_month["revenue"]
    )

    # Top category
    if not categories.empty:

        top_category = categories.iloc[0]

        logger.info(
            "Top category: %s | revenue=%.2f",
            top_category["category"],
            top_category["revenue"]
        )

        # Number of categories needed to reach 80%
        pareto_80 = pareto[
            pareto["cumulative_revenue_pct"] <= 80
        ]

        logger.info(
            "Categories within first 80%% cumulative revenue: %d",
            len(pareto_80)
        )

    # Top seller
    if not sellers.empty:

        top_seller = sellers.iloc[0]

        logger.info(
            "Top seller: %s | revenue=%.2f",
            top_seller["seller_id"],
            top_seller["revenue"]
        )

    # Top state
    if not states.empty:

        top_state = states.iloc[0]

        logger.info(
            "Top customer state: %s | revenue=%.2f",
            top_state["state"],
            top_state["revenue"]
        )

    logger.info("=" * 70)


# ---------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete business-performance analysis."""

    logger.info("=" * 70)
    logger.info("OLIST BUSINESS PERFORMANCE ANALYSIS")
    logger.info("=" * 70)

    connection = get_connection()

    try:

        # -------------------------------------------------------------
        # Monthly revenue
        # -------------------------------------------------------------

        logger.info("Calculating monthly revenue...")

        monthly = monthly_revenue(
            connection
        )

        monthly = add_monthly_growth(
            monthly
        )

        logger.info(
            "Monthly revenue rows: %d",
            len(monthly)
        )

        # -------------------------------------------------------------
        # Category performance
        # -------------------------------------------------------------

        logger.info(
            "Calculating category performance..."
        )

        categories = category_performance(
            connection
        )

        logger.info(
            "Categories: %d",
            len(categories)
        )

        # -------------------------------------------------------------
        # Pareto
        # -------------------------------------------------------------

        logger.info(
            "Calculating category Pareto analysis..."
        )

        pareto = pareto_analysis(
            categories
        )

        # -------------------------------------------------------------
        # Seller performance
        # -------------------------------------------------------------

        logger.info(
            "Calculating seller performance..."
        )

        sellers = seller_performance(
            connection
        )

        logger.info(
            "Sellers: %d",
            len(sellers)
        )

        # -------------------------------------------------------------
        # State performance
        # -------------------------------------------------------------

        logger.info(
            "Calculating state performance..."
        )

        states = state_performance(
            connection
        )

        logger.info(
            "States: %d",
            len(states)
        )

        # -------------------------------------------------------------
        # Save
        # -------------------------------------------------------------

        save_results(
            connection,
            monthly,
            categories,
            pareto,
            sellers,
            states
        )

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------

        print_summary(
            monthly,
            categories,
            pareto,
            sellers,
            states
        )

        # -------------------------------------------------------------
        # Display important tables
        # -------------------------------------------------------------

        print("\n" + "=" * 70)
        print("MONTHLY REVENUE")
        print("=" * 70)

        print(
            monthly.to_string(
                index=False
            )
        )

        print("\n" + "=" * 70)
        print("TOP 10 CATEGORIES")
        print("=" * 70)

        print(
            top_categories(
                categories,
                10
            ).to_string(
                index=False
            )
        )

        print("\n" + "=" * 70)
        print("TOP 20 SELLERS")
        print("=" * 70)

        print(
            top_sellers(
                sellers,
                20
            ).to_string(
                index=False
            )
        )

        print("\n" + "=" * 70)
        print("TOP 10 STATES")
        print("=" * 70)

        print(
            top_states(
                states,
                10
            ).to_string(
                index=False
            )
        )

        logger.info(
            "Business-performance analysis completed successfully."
        )

    finally:
        connection.close()


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
