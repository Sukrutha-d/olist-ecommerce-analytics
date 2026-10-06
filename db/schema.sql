-- ============================================================
-- OLIST ANALYTICS DATA WAREHOUSE
-- Star Schema
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- DROP EXISTING WAREHOUSE TABLES
-- ============================================================

DROP TABLE IF EXISTS fact_orders;

DROP TABLE IF EXISTS dim_customer;
DROP TABLE IF EXISTS dim_product;
DROP TABLE IF EXISTS dim_seller;
DROP TABLE IF EXISTS dim_date;


-- ============================================================
-- DIMENSION: CUSTOMER
-- One row per unique customer identity
-- ============================================================

CREATE TABLE dim_customer (
    customer_key INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_unique_id TEXT NOT NULL UNIQUE,
    city TEXT,
    state TEXT
);


-- ============================================================
-- DIMENSION: PRODUCT
-- One row per product
-- ============================================================

CREATE TABLE dim_product (
    product_key INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL UNIQUE,
    category TEXT,
    weight_g REAL,
    length_cm REAL,
    height_cm REAL,
    width_cm REAL
);


-- ============================================================
-- DIMENSION: SELLER
-- One row per seller
-- ============================================================

CREATE TABLE dim_seller (
    seller_key INTEGER PRIMARY KEY AUTOINCREMENT,
    seller_id TEXT NOT NULL UNIQUE,
    city TEXT,
    state TEXT
);


-- ============================================================
-- DIMENSION: DATE
-- One row per calendar date
-- ============================================================

CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,
    date TEXT NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL
);


-- ============================================================
-- FACT: ORDERS
--
-- Grain:
-- One row = one product/item within an order
--
-- IMPORTANT:
-- payment_value is an order-level aggregated value.
-- It must NOT be blindly summed across fact rows because
-- an order can contain multiple items.
-- ============================================================

CREATE TABLE fact_orders (
    order_id TEXT NOT NULL,
    order_item_id INTEGER NOT NULL,

    customer_key INTEGER NOT NULL,
    product_key INTEGER NOT NULL,
    seller_key INTEGER NOT NULL,
    date_key INTEGER NOT NULL,

    price REAL NOT NULL,
    freight_value REAL NOT NULL,

    payment_value REAL,

    review_score INTEGER,

    delivery_days REAL,
    estimated_delivery_days REAL,
    delay_days REAL,
    is_late INTEGER,

    PRIMARY KEY (order_id, order_item_id),

    FOREIGN KEY (customer_key)
        REFERENCES dim_customer(customer_key),

    FOREIGN KEY (product_key)
        REFERENCES dim_product(product_key),

    FOREIGN KEY (seller_key)
        REFERENCES dim_seller(seller_key),

    FOREIGN KEY (date_key)
        REFERENCES dim_date(date_key)
);


-- ============================================================
-- INDEXES
-- Improve joins and analytical queries
-- ============================================================

CREATE INDEX idx_fact_orders_customer
    ON fact_orders(customer_key);

CREATE INDEX idx_fact_orders_product
    ON fact_orders(product_key);

CREATE INDEX idx_fact_orders_seller
    ON fact_orders(seller_key);

CREATE INDEX idx_fact_orders_date
    ON fact_orders(date_key);

CREATE INDEX idx_fact_orders_order
    ON fact_orders(order_id);

CREATE INDEX idx_dim_customer_unique_id
    ON dim_customer(customer_unique_id);

CREATE INDEX idx_dim_product_category
    ON dim_product(category);

CREATE INDEX idx_dim_seller_state
    ON dim_seller(state);

CREATE INDEX idx_dim_date_date
    ON dim_date(date);
