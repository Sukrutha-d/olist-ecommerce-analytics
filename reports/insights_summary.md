cat > reports/insights_summary.md <<'EOF'
# Olist E-Commerce Analytics — Business Insights

## Executive Summary

The Olist Brazilian E-Commerce dataset contains approximately 99K
orders and 96K unique customers. The analysis shows strong overall
sales performance but very low customer retention. Delivery
performance is strongly associated with customer satisfaction, while
revenue is highly concentrated across a small number of categories
and states.

The main business opportunity is therefore not simply acquiring more
customers, but improving second-order conversion and reducing delivery
delays.

---

## 1. Customer Retention

The customer analysis identified:

- 96,096 unique customers.
- 93,099 customers placed only one order.
- 2,997 customers placed multiple orders.
- Overall repeat-customer rate: **3.12%**.
- Approximately **96.88%** of customers are one-time customers.
- Median time to the second order: **27.9 days**.
- Average time to the second order: **80.3 days**.

### Business implication

Customer retention is the largest opportunity identified by the
analysis.

A large majority of customers do not return after their first
purchase. Marketing and CRM efforts should therefore focus on
converting first-time buyers into second-time buyers.

Potential actions include:

- Post-purchase campaigns.
- Personalized product recommendations.
- Category-specific cross-selling.
- Second-order discounts.
- Follow-up communication shortly after delivery.

The median time to a second purchase of approximately 28 days
suggests that the first month after purchase is an important period
for retention campaigns.

---

## 2. Customer Segmentation

RFM analysis was performed using:

- Recency
- Frequency
- Monetary value

Customer identity was based on `customer_unique_id`.

The final K-Means segmentation produced four interpretable groups:

| Segment | Customers | Share |
|---|---:|---:|
| High-Value Inactive | 35,957 | 37.42% |
| Recent One-Time | 25,962 | 27.02% |
| Repeat Customers | 2,997 | 3.12% |
| Low-Value Inactive | 31,180 | 32.45% |

### Business implication

The **High-Value Inactive** group is particularly important because
these customers have previously demonstrated relatively high monetary
value but are no longer recent purchasers.

Recommended priorities:

1. Reactivate high-value inactive customers.
2. Convert recent one-time customers into repeat buyers.
3. Protect existing repeat customers.
4. Avoid spending equally on all inactive low-value customers.

---

## 3. Delivery Performance and Customer Satisfaction

Delivery analysis examined 95,830 delivered orders with review
information.

Average review scores decreased substantially as delivery delays
increased:

| Delay | Average Review |
|---|---:|
| Early / On Time | 4.29 |
| 1–3 days late | 3.77 |
| 4–7 days late | 2.32 |
| 8–14 days late | 1.74 |
| 15+ days late | 1.71 |

The Spearman correlation between delivery delay and review score was:

**rho = -0.176, p < 0.001**

A Mann-Whitney test also showed a statistically significant difference
between late and on-time orders:

- Late-order mean review: **2.57**
- On-time-order mean review: **4.29**
- Rank-biserial effect size: approximately **-0.554**
- p < 0.001

### Business implication

Delivery delays are strongly associated with poorer customer
satisfaction.

The largest deterioration occurs once delays exceed approximately
four days.

Operational priorities should therefore include:

- Monitoring sellers with high late-delivery rates.
- Improving carrier performance.
- Identifying problematic origin-destination combinations.
- Early intervention on orders approaching estimated delivery dates.

These findings show association, not proof of causation.

---

## 4. First-Order Delivery and Repeat Purchasing

Customers whose first order was delivered on time had a repeat rate of:

**3.02%**

Customers whose first order was late had a repeat rate of:

**2.48%**

A chi-square test produced:

**p = 0.0094**

### Business implication

Late delivery on the first purchase is associated with a lower
probability of subsequent purchasing.

This strengthens the business case for prioritizing delivery quality
for new customers.

---

## 5. Revenue Performance

Delivered product revenue was approximately:

**13.22 million**

Total delivered orders:

**96,478**

Total freight:

**2.20 million**

The strongest revenue month in the observed period was:

**November 2017 — approximately 987,765**

The lowest meaningful early-period month was affected by the sparse
beginning of the dataset and should not be interpreted as normal
business performance.

---

## 6. Category Performance

The highest-revenue categories were:

| Category | Revenue |
|---|---:|
| Health & Beauty | 1,233,131.72 |
| Watches & Gifts | 1,166,176.98 |
| Bed Bath Table | 1,023,434.76 |
| Sports & Leisure | 954,852.55 |
| Computers & Accessories | 888,724.61 |

The Pareto analysis found that approximately **17 categories**
accounted for the first 80% of cumulative category revenue.

### Business implication

Revenue is concentrated among a relatively small set of categories.

Management should prioritize:

- Inventory availability for high-revenue categories.
- Seller quality within these categories.
- Delivery performance for high-value categories.
- Cross-selling between complementary categories.

---

## 7. Geographic Performance

São Paulo was the strongest customer state by revenue:

**SP revenue: approximately 5.07 million**

Other major states included:

- RJ: approximately 1.76 million
- MG: approximately 1.55 million
- RS: approximately 0.73 million
- PR: approximately 0.67 million

### Business implication

São Paulo represents the largest market and should receive particular
attention for:

- Inventory positioning.
- Seller coverage.
- Delivery optimization.
- Customer retention campaigns.

However, high-revenue states should not be the only focus. Smaller
states with high delivery-delay rates may represent operational
improvement opportunities.

---

## 8. Seller Performance

The highest-revenue seller generated approximately:

**226,987.93**

The top seller was:

`4869f7a5dfa277a7dca6462dcf3b52b2`

The analysis also showed that seller performance is highly
heterogeneous, with a relatively small number of sellers generating
substantial revenue.

### Business implication

Seller management should combine both revenue and operational
performance.

A high-revenue seller with poor delivery performance may have a larger
business impact than a low-volume seller with the same delay rate.

A useful seller scorecard should therefore combine:

- Revenue
- Orders
- Average order/item value
- Freight
- Late-delivery rate
- Customer review score

---

## 9. Revenue Forecast

Four forecasting approaches were compared:

| Model | MAPE |
|---|---:|
| Naive | **10.60%** |
| ARIMA | 12.23% |
| SARIMA | 27.29% |
| Seasonal Naive | 44.76% |

The **Naive model** achieved the lowest validation MAPE and was
therefore selected.

The six-month forecast is approximately:

**838,576.64 per month**

for:

- September 2018
- October 2018
- November 2018
- December 2018
- January 2019
- February 2019

### Important limitation

The dataset contains only a relatively short monthly history.
Therefore, complex seasonal models such as SARIMA have limited
statistical support.

The fact that the naive baseline outperformed ARIMA and SARIMA is
therefore useful evidence that model complexity should not be assumed
to improve forecasting accuracy.

---

## 10. Recommended Business Actions

### Priority 1 — Increase second-order conversion

The repeat rate of only 3.12% is the largest customer-growth
opportunity.

Focus on:

- Recent one-time customers.
- Customers with high first-order value.
- Customers approaching the typical second-purchase window.

---

### Priority 2 — Improve delivery reliability

Delivery delays are strongly associated with lower review scores.

Focus on:

- High-delay sellers.
- States with elevated late-delivery rates.
- Orders approaching estimated delivery dates.
- New customers whose first order is at risk of being late.

---

### Priority 3 — Reactivate high-value inactive customers

The High-Value Inactive segment contains approximately 35,957
customers.

Target these customers with:

- Personalized offers.
- Relevant category recommendations.
- Limited-time incentives.
- Re-engagement campaigns.

---

### Priority 4 — Protect high-performing categories

Health & Beauty, Watches & Gifts, Bed Bath Table, Sports & Leisure,
and Computers & Accessories are major revenue contributors.

Maintain strong:

- Inventory availability.
- Seller quality.
- Delivery performance.
- Customer support.

---

### Priority 5 — Optimize the largest markets

São Paulo is the dominant revenue-generating customer state.

Operational optimization in high-volume states can have a larger
absolute business impact than optimizing very small markets.

---

## Final Conclusion

The analysis reveals a business with strong transaction volume but
very low customer retention.

The strongest combined opportunity is:

**Improve the first-order experience → convert customers into a second
purchase → retain high-value customers.**

Delivery reliability is a particularly important part of this
strategy because delayed orders are associated with substantially
lower review scores and a lower subsequent purchase rate.

Revenue is concentrated across a relatively small number of categories,
sellers and states, making targeted operational and marketing
interventions more practical than treating the entire customer base
uniformly.
EOF
