"""
Investigate suspicious date records before cleaning.
"""

import sqlite3
import pandas as pd

from config import DB_PATH


def get_connection():
    return sqlite3.connect(DB_PATH)


def show_carrier_before_purchase(conn):
    """Show orders where carrier delivery is before purchase."""

    query = """
        SELECT
            order_id,
            customer_id,
            order_status,
            order_purchase_timestamp,
            order_approved_at,
            order_delivered_carrier_date,
            order_delivered_customer_date,
            order_estimated_delivery_date
        FROM raw_orders
        WHERE order_delivered_carrier_date IS NOT NULL
          AND datetime(order_delivered_carrier_date)
              < datetime(order_purchase_timestamp)
        ORDER BY order_purchase_timestamp
    """

    df = pd.read_sql_query(query, conn)

    print("\n" + "=" * 100)
    print("CARRIER DELIVERY BEFORE PURCHASE")
    print("=" * 100)

    print(f"Number of problematic orders: {len(df)}")
    print()

    if not df.empty:
        print(df.to_string(index=False))


def show_customer_before_carrier(conn):
    """Show orders where customer delivery is before carrier delivery."""

    query = """
        SELECT
            order_id,
            customer_id,
            order_status,
            order_purchase_timestamp,
            order_approved_at,
            order_delivered_carrier_date,
            order_delivered_customer_date,
            order_estimated_delivery_date
        FROM raw_orders
        WHERE order_delivered_customer_date IS NOT NULL
          AND order_delivered_carrier_date IS NOT NULL
          AND datetime(order_delivered_customer_date)
              < datetime(order_delivered_carrier_date)
        ORDER BY order_purchase_timestamp
    """

    df = pd.read_sql_query(query, conn)

    print("\n" + "=" * 100)
    print("CUSTOMER DELIVERY BEFORE CARRIER DELIVERY")
    print("=" * 100)

    print(f"Number of problematic orders: {len(df)}")
    print()

    if not df.empty:
        print(df.to_string(index=False))


def main():
    conn = get_connection()

    try:
        show_carrier_before_purchase(conn)
        show_customer_before_carrier(conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
