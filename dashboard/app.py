"""
Olist Analytics Dashboard

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ================================================================
# CONFIG
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = (
    PROJECT_ROOT
    / "db"
    / "olist.db"
)


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="Olist E-Commerce Analytics",
    page_icon="📊",
    layout="wide"
)


# ================================================================
# DATABASE
# ================================================================

@st.cache_resource
def get_connection():
    """Create SQLite connection."""

    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


connection = get_connection()


# ================================================================
# DATA LOADERS
# ================================================================

@st.cache_data
def load_monthly():
    """Load monthly business performance."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM monthly_business_performance
        ORDER BY month
        """,
        connection
    )


@st.cache_data
def load_categories():
    """Load category performance."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM category_performance
        ORDER BY revenue DESC
        """,
        connection
    )


@st.cache_data
def load_sellers():
    """Load seller performance."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM seller_performance
        ORDER BY revenue DESC
        """,
        connection
    )


@st.cache_data
def load_states():
    """Load state business performance."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM state_business_performance
        ORDER BY revenue DESC
        """,
        connection
    )


@st.cache_data
def load_rfm():
    """Load RFM segmentation."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM rfm_segments
        """,
        connection
    )


@st.cache_data
def load_customer_features():
    """Load customer-level features."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM customer_features
        """,
        connection
    )


@st.cache_data
def load_delivery_buckets():
    """Load delivery/review analysis."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM delivery_review_buckets
        ORDER BY
            CASE delay_bucket
                WHEN 'Early / On Time' THEN 1
                WHEN '1-3 days late' THEN 2
                WHEN '4-7 days late' THEN 3
                WHEN '8-14 days late' THEN 4
                WHEN '15+ days late' THEN 5
                ELSE 6
            END
        """,
        connection
    )


@st.cache_data
def load_state_delivery():
    """Load state delivery performance."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM state_delivery_performance
        ORDER BY late_rate_pct DESC
        """,
        connection
    )


@st.cache_data
def load_forecast():
    """Load revenue forecast."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM revenue_forecast
        ORDER BY month
        """,
        connection
    )


@st.cache_data
def load_model_comparison():
    """Load forecasting model comparison."""

    return pd.read_sql_query(
        """
        SELECT *
        FROM forecast_model_comparison
        ORDER BY MAPE
        """,
        connection
    )


# ================================================================
# LOAD DATA
# ================================================================

monthly = load_monthly()
categories = load_categories()
sellers = load_sellers()
states = load_states()

rfm = load_rfm()
customer_features = load_customer_features()

delivery_buckets = load_delivery_buckets()
state_delivery = load_state_delivery()

forecast = load_forecast()
model_comparison = load_model_comparison()


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.title("📊 Olist Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Customers",
        "Delivery",
        "Sellers",
        "Forecast"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Brazilian E-Commerce Analytics"
)

st.sidebar.caption(
    "Olist Public Dataset"
)


# ================================================================
# HELPER
# ================================================================

def money(value: float) -> str:
    """Format monetary value."""

    return f"{value:,.2f}"


# ================================================================
# OVERVIEW
# ================================================================

if page == "Overview":

    st.title(
        "🛒 Olist E-Commerce Analytics"
    )

    st.markdown(
        """
        **Business performance overview**

        This dashboard analyzes customer behavior, sales,
        delivery performance, sellers and revenue forecasts.
        """
    )

    # ------------------------------------------------------------
    # KPIs
    # ------------------------------------------------------------

    total_revenue = monthly["revenue"].sum()
    total_orders = monthly["orders"].sum()
    total_customers = monthly["customers"].sum()

    repeat_rate = (
        customer_features["repeat_customer"]
        .mean()
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Delivered Revenue",
        money(total_revenue)
    )

    col2.metric(
        "Delivered Orders",
        f"{int(total_orders):,}"
    )

    col3.metric(
        "Customers",
        f"{int(total_customers):,}"
    )

    col4.metric(
        "Repeat Rate",
        f"{repeat_rate:.2f}%"
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Revenue trend
    # ------------------------------------------------------------

    st.subheader(
        "Monthly Revenue"
    )

    fig = px.line(
        monthly,
        x="month",
        y="revenue",
        markers=True,
        title="Monthly Delivered Revenue"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Top Categories"
        )

        top = categories.head(10)

        fig = px.bar(
            top,
            x="revenue",
            y="category",
            orientation="h",
            title="Top 10 Categories by Revenue"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader(
            "Top States"
        )

        top_states = states.head(10)

        fig = px.bar(
            top_states,
            x="revenue",
            y="state",
            orientation="h",
            title="Top 10 States by Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ================================================================
# CUSTOMERS
# ================================================================

elif page == "Customers":

    st.title(
        "👥 Customer Analytics"
    )

    # ------------------------------------------------------------
    # Customer KPIs
    # ------------------------------------------------------------

    total_customers = len(
        customer_features
    )

    repeat_customers = int(
        customer_features[
            "repeat_customer"
        ].sum()
    )

    one_time = (
        total_customers
        - repeat_customers
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Unique Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Repeat Customers",
        f"{repeat_customers:,}"
    )

    col3.metric(
        "One-Time Customers",
        f"{one_time:,}"
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Repeat vs one-time
    # ------------------------------------------------------------

    repeat_df = pd.DataFrame(
        {
            "Customer Type": [
                "One-Time",
                "Repeat"
            ],
            "Customers": [
                one_time,
                repeat_customers
            ]
        }
    )

    fig = px.pie(
        repeat_df,
        names="Customer Type",
        values="Customers",
        title="Customer Retention Profile"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # RFM segments
    # ------------------------------------------------------------

    st.subheader(
        "RFM Customer Segments"
    )

    if "segment" in rfm.columns:

        segment_counts = (
            rfm["segment"]
            .value_counts()
            .reset_index()
        )

        segment_counts.columns = [
            "segment",
            "customers"
        ]

        fig = px.bar(
            segment_counts,
            x="customers",
            y="segment",
            orientation="h",
            title="Customers by RFM Segment"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ------------------------------------------------------------
    # Frequency
    # ------------------------------------------------------------

    st.subheader(
        "Purchase Frequency"
    )

    frequency = (
        customer_features[
            "frequency"
        ]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    frequency.columns = [
        "orders",
        "customers"
    ]

    fig = px.bar(
        frequency,
        x="orders",
        y="customers",
        title="Customer Purchase Frequency"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ================================================================
# DELIVERY
# ================================================================

elif page == "Delivery":

    st.title(
        "🚚 Delivery Performance"
    )

    st.markdown(
        """
        Delivery analysis examines how delays relate to customer
        review scores and repeat purchasing.
        """
    )

    # ------------------------------------------------------------
    # Delay / review
    # ------------------------------------------------------------

    st.subheader(
        "Delivery Delay vs Review Score"
    )

    fig = px.bar(
        delivery_buckets,
        x="delay_bucket",
        y="avg_review",
        title="Average Review Score by Delivery Delay"
    )

    fig.update_layout(
        xaxis_title="Delay Bucket",
        yaxis_title="Average Review Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Delivery table
    # ------------------------------------------------------------

    st.subheader(
        "Delay Bucket Performance"
    )

    st.dataframe(
        delivery_buckets,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # State delivery
    # ------------------------------------------------------------

    st.subheader(
        "State Delivery Performance"
    )

    if not state_delivery.empty:

        fig = px.bar(
            state_delivery.head(15),
            x="late_rate_pct",
            y="state",
            orientation="h",
            title="States with Highest Late-Delivery Rate"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "total ascending"
            },
            xaxis_title="Late Rate (%)",
            yaxis_title="State"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ------------------------------------------------------------
    # Statistical finding
    # ------------------------------------------------------------

    st.info(
        """
        Statistical analysis found a significant negative association
        between delivery delay and review score. Late first orders also
        showed a lower repeat-purchase rate than on-time first orders.
        Association does not by itself establish causation.
        """
    )


# ================================================================
# SELLERS
# ================================================================

elif page == "Sellers":

    st.title(
        "🏪 Seller Analytics"
    )

    # ------------------------------------------------------------
    # Top sellers
    # ------------------------------------------------------------

    st.subheader(
        "Top Sellers by Revenue"
    )

    top_sellers = sellers.head(20)

    fig = px.bar(
        top_sellers,
        x="revenue",
        y="seller_id",
        orientation="h",
        title="Top 20 Sellers"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Seller table
    # ------------------------------------------------------------

    st.subheader(
        "Seller Performance"
    )

    display_columns = [
        "seller_id",
        "city",
        "state",
        "orders",
        "revenue",
        "freight",
        "avg_item_value"
    ]

    st.dataframe(
        sellers[
            display_columns
        ].head(50),
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Seller state distribution
    # ------------------------------------------------------------

    seller_state = (
        sellers
        .groupby("state", as_index=False)
        .agg(
            sellers=(
                "seller_id",
                "nunique"
            ),
            revenue=(
                "revenue",
                "sum"
            )
        )
        .sort_values(
            "revenue",
            ascending=False
        )
    )

    st.subheader(
        "Seller Revenue by State"
    )

    fig = px.bar(
        seller_state.head(15),
        x="state",
        y="revenue",
        title="Seller Revenue by State"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ================================================================
# FORECAST
# ================================================================

elif page == "Forecast":

    st.title(
        "📈 Revenue Forecast"
    )

    # ------------------------------------------------------------
    # Best model
    # ------------------------------------------------------------

    best_model = (
        model_comparison
        .iloc[0]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Selected Model",
        best_model["model"]
    )

    col2.metric(
        "MAPE",
        f"{best_model['MAPE']:.2f}%"
    )

    col3.metric(
        "RMSE",
        money(best_model["RMSE"])
    )

    st.markdown("---")

    # ------------------------------------------------------------
    # Historical + forecast
    # ------------------------------------------------------------

    historical = monthly.copy()

    historical["type"] = "Historical"

    historical_plot = historical[
        [
            "month",
            "revenue",
            "type"
        ]
    ].rename(
        columns={
            "revenue": "value"
        }
    )

    future_plot = forecast.copy()

    future_plot["type"] = "Forecast"

    future_plot = future_plot[
        [
            "month",
            "forecast_revenue",
            "type"
        ]
    ].rename(
        columns={
            "forecast_revenue": "value"
        }
    )

    combined = pd.concat(
        [
            historical_plot,
            future_plot
        ],
        ignore_index=True
    )

    fig = px.line(
        combined,
        x="month",
        y="value",
        color="type",
        markers=True,
        title="Historical Revenue and 6-Month Forecast"
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------

    st.subheader(
        "Forecast Model Comparison"
    )

    st.dataframe(
        model_comparison,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Forecast table
    # ------------------------------------------------------------

    st.subheader(
        "Next 6 Months"
    )

    st.dataframe(
        forecast,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------

    st.info(
        f"""
        The {best_model['model']} model was selected because it achieved
        the lowest validation MAPE of {best_model['MAPE']:.2f}%.
        More complex models were tested but did not outperform this
        baseline on the available historical data.
        """
    )


# ================================================================
# FOOTER
# ================================================================

st.markdown("---")

st.caption(
    "Olist Brazilian E-Commerce Analytics | "
    "Data Engineering + SQL + RFM + Delivery Analytics + Forecasting"
)
