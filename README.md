# Olist Brazilian E-Commerce Analytics

An end-to-end data analytics and machine learning project using the
Olist Brazilian E-Commerce Public Dataset.

The project combines data engineering, SQL analytics, customer
segmentation, cohort analysis, delivery-performance analysis and
revenue forecasting into an interactive Streamlit dashboard.

---

## Business Questions

The project answers:

1. Who are the most valuable customers?
2. What percentage of customers return?
3. Which customer segments should be prioritized?
4. How does delivery delay affect customer satisfaction?
5. Does a late first order relate to repeat purchasing?
6. Which product categories and sellers generate the most revenue?
7. Which Brazilian states contribute the most revenue?
8. What is the expected revenue for the next 3–6 months?

---

## Dataset

The project uses the Olist Brazilian E-Commerce Public Dataset.

The dataset contains:

- Orders
- Order items
- Payments
- Reviews
- Customers
- Sellers
- Products
- Geolocation
- Product-category translations

---

## Architecture

text
Raw CSV Files
     |
     v
Data Ingestion
     |
     v
Data Profiling
     |
     v
Data Quality Checks
     |
     v
Data Cleaning
     |
     v
SQLite Warehouse
     |
     +-----------------------------+
     |                             |
     v                             v
Feature Engineering          SQL Analytics
     |                             |
     +-------------+---------------+
                   |
                   v
          Customer Analytics
          - RFM
          - K-Means
          - Cohorts
          - Retention
                   |
                   v
          Delivery Analytics
          - Delay buckets
          - Reviews
          - Repeat purchase
                   |
                   v
             Forecasting
          - Naive
          - Seasonal Naive
          - ARIMA
          - SARIMA
                   |
                   v
          Streamlit Dashboard
Data Warehouse
A star schema is used.
Fact
fact_orders
Grain:
One row per order item.

Measures include:
- price
- freight_value
- payment_value
- review_score
- delivery_days
- delay_days
- is_late
Dimensions
- dim_customer
- dim_product
- dim_seller
- dim_date
Data Quality
The pipeline validates:
- Primary-key uniqueness
- Foreign-key integrity
- Missing values
- Date consistency
- Negative prices
- Negative freight
- Invalid review scores
- Invalid payment installments
Impossible delivery timestamps were handled during cleaning rather
than deleting complete orders.
Customer Analytics
Customer identity is based on:
customer_unique_id
rather than the order-level customer_id.
This prevents the same real customer from being treated as multiple
customers when they place multiple orders.
RFM
The project calculates:
- Recency
- Frequency
- Monetary value
Both rule-based segmentation and K-Means clustering are implemented.
Final K-Means segments:
- High-Value Inactive
- Recent One-Time
- Repeat Customers
- Low-Value Inactive
Retention
Cohort analysis uses the customer's first purchase month.
The analysis calculates:
- Monthly retention
- Repeat customer rate
- Time to second order
Key result:
- 96.88% of customers are one-time customers.
- 3.12% are repeat customers.
- Median time to a second order is approximately 27.9 days.
Delivery Analysis
Delivery delay is analyzed against customer review scores.
Delay buckets:
- Early / On Time
- 1–3 days late
- 4–7 days late
- 8–14 days late
- 15+ days late
Key finding:
Delivery delay has a statistically significant negative association
with review score.
The analysis also compares repeat purchasing after on-time versus
late first orders.
Association is reported as association rather than causation.
Business Performance
Delivered product revenue is used as the primary revenue measure.
Freight is reported separately.
Key findings include:
- Total delivered revenue: approximately 13.22 million
- Total delivered orders: 96,478
- Health & Beauty is the highest-revenue category.
- São Paulo is the highest-revenue customer state.
- 17 categories account for approximately the first 80% of cumulative
  category revenue.
Forecasting
Monthly delivered revenue is forecast using:
- Naive
- Seasonal Naive
- ARIMA
- SARIMA
Models are evaluated using a chronological train/test split.
The final model is selected using validation MAPE.
Final selected model:
Naive Forecast
Validation MAPE:
10.60%
The model produces a six-month forecast.
The simple baseline outperformed the more complex models on the
available historical data, demonstrating why model selection should be
based on out-of-sample performance rather than model complexity.
Dashboard
The Streamlit dashboard contains:
Overview
- Revenue
- Orders
- Customers
- Repeat rate
- Monthly revenue
- Categories
- States
Customers
- One-time vs repeat customers
- RFM segments
- Purchase frequency
Delivery
- Delay buckets
- Review scores
- State delivery performance
Sellers
- Top sellers
- Seller revenue
- Seller states
Forecast
- Historical revenue
- Six-month forecast
- Model comparison
- MAPE
- RMSE
Running the Project
Create the environment:
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Place the Olist CSV files inside:
data/raw/

Run the pipeline modules:
python3 -m src.load_data
python3 -m src.profile_data
python3 -m src.data_quality
python3 -m src.clean_data
python3 -m src.build_warehouse
python3 -m src.features
python3 -m src.rfm
python3 -m src.rfm_segments
python3 -m src.cohort_analysis
python3 -m src.cohort_heatmap
python3 -m src.delivery_analysis
python3 -m src.business_performance
python3 -m src.forecast

Run tests:
pytest -q

Start the dashboard:
streamlit run dashboard/app.py

Technologies
- Python
- Pandas
- NumPy
- SQLite
- SQL
- Scikit-learn
- SciPy
- Statsmodels
- Plotly
- Streamlit
- Matplotlib
- Seaborn
- Pytest
Project Highlights
This project demonstrates:
- Data ingestion
- Data cleaning
- Data quality engineering
- Relational data modeling
- Star-schema warehouse design
- SQL analytics
- Feature engineering
- RFM segmentation
- K-Means clustering
- Cohort analysis
- Statistical hypothesis testing
- Time-series forecasting
- Interactive dashboard development
Author
Sukrutha D.
B.Tech Computer Science and Engineering
PES University
